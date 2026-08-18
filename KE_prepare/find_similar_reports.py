"""Step 2 of 2 for the MvKeTR-style knowledge base: compute each patient's own
per-organ IMAGE embedding from a finetuned checkpoint, then retrieve the top-k most
similar OTHER patients' report TEXT embeddings (from compute_text_embeddings.py's
organ_report_embeddings.npz) for each organ.

Run this only once fine-tuning has actually converged - unlike the text side, image
embeddings for the 6 non-frozen organs depend entirely on how well-trained
vision_projs is, so running this against an early/undertrained checkpoint and
re-running it later against the final one would just throw away the (expensive) work.

Retrieval is cross-modal, not text-to-text: at real report-generation time the
current patient's own report doesn't exist yet, so a patient's own report text can't
be the retrieval query - only their CT volume can. So for each organ, the query is
that patient's own organ IMAGE embedding (via the model's eval-time windowing -
eval_finetune.py's center_crop + forward_test_win, same ROI-pool -> vision_proj path
used everywhere else in this project), searched against every OTHER patient's organ
TEXT embedding. This mirrors the original repo, where clip_memory (retrieved via
CT-CLIP) is text embeddings even though retrieval itself is presumably image-anchored.

Output: organ_annotation.json - {"train": [ {"id", "image_path", "<organ>",
"<organ>_indices" for each organ in ORGANS} ]}. "<organ>_indices" are row indices
into organ_report_embeddings.npz's feats/patient_ids for that organ's top-k nearest
TEXT neighbors to this patient's own organ IMAGE embedding, by cosine similarity,
excluding the patient's own row. -1 means no valid image embedding was available for
that (patient, organ) pair (e.g. no mask for that organ), so there's no query to
retrieve with.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from monai import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finetune import CROP_SIZE, DATA_ROOT, ORGANS, PATCH_SIZE, PREPROCESSED_IMAGE_ROOT, PREPROCESSED_MASK_ROOT, RADGENOME_CSV, build_organ_captions
from eval_finetune import DEFAULT_OUTPUT_DIR, _default_checkpoint, build_eval_model, center_crop
from compute_text_embeddings import clean_caption


@torch.inference_mode()
def compute_image_embeddings(model, patient_ids, device):
    """Per organ, per patient, a single pooled 256-dim image embedding: center_crop
    around that organ's own mask, pad, then the same ROI-pool -> attention ->
    vision_proj path forward_test_win uses everywhere else (blip_pretrain.py:501-524).
    One ViT forward per organ present (not per patient) - each organ gets its own
    crop window, so this is up to 9x the cost of a single per-patient forward, same
    as eval_finetune.py's own evaluation loop already does per validation sample.

    organ_logits={} and text_feat_dict={} passed to forward_test_win: we only want
    the image embedding it writes into organ_feat_dict, not its text-comparison
    logits, and an empty organ_logits dict makes that inner loop a no-op.
    """
    loader = transforms.Compose([
        transforms.LoadImaged(keys=["image", "label"], image_only=True, ensure_channel_first=True),
    ])
    pad_func = transforms.DivisiblePadd(
        keys=["image", "label"], k=PATCH_SIZE, mode="constant", constant_values=0, method="end",
    )

    n = len(patient_ids)
    feats = np.zeros((n, len(ORGANS), 256), dtype=np.float32)
    valid = np.zeros((n, len(ORGANS)), dtype=bool)

    for i, pid in enumerate(tqdm(patient_ids, desc="image embed")):
        file_name = f"{pid}.nii.gz"
        data = loader({
            "image": os.path.join(PREPROCESSED_IMAGE_ROOT, file_name),
            "label": os.path.join(PREPROCESSED_MASK_ROOT, file_name),
        })
        image = data["image"].as_tensor()[None].to(device)
        mask = data["label"].as_tensor()[None].to(device)

        whole_organ_sizes = {
            organ: torch.eq(mask, ORGANS.index(organ) + 1).sum().item() for organ in ORGANS
        }
        organ_feat_dict = {}
        for organ_id, organ in enumerate(ORGANS):
            if whole_organ_sizes[organ] == 0:
                continue  # this patient's preprocessed mask doesn't have this organ at all

            window_patch, window_mask = center_crop(image, torch.eq(mask, organ_id + 1), crop_size=CROP_SIZE)
            window_mask = window_mask.float()
            window_mask[window_mask == 1] = organ_id + 1
            pad_data = pad_func({"image": window_patch[0], "label": window_mask[0]})
            window_patch, window_mask = pad_data["image"], pad_data["label"]

            model.forward_test_win(
                window_patch[None], window_mask[None], {}, [organ],
                {}, organ_feat_dict, whole_organ_sizes, skip_organ=organ_id,
            )
            if organ in organ_feat_dict:
                feats[i, organ_id, :] = np.array(organ_feat_dict[organ][0], dtype=np.float32)
                valid[i, organ_id] = True

    return feats, valid


def compute_top_k_indices(image_feats, image_valid, text_feats, top_k, device):
    """Per organ: cosine similarity (plain dot product - both sides are already
    L2-normalized) of every patient's own IMAGE embedding against every OTHER
    patient's TEXT embedding, top_k nearest excluding the patient's own row. Full
    N x N similarity matrix, one organ at a time (~2.3GB for N=24116 in float32) -
    small enough to do directly, no approximate search needed at this scale.

    Patients missing this organ's mask (image_valid False) get an all -1 result -
    there's no image query to retrieve with for them.
    """
    n = text_feats.shape[0]
    all_indices = np.full((n, len(ORGANS), top_k), -1, dtype=np.int64)
    for organ_idx, organ in enumerate(tqdm(ORGANS, desc="top-k neighbors")):
        img = torch.from_numpy(image_feats[:, organ_idx, :]).to(device)
        txt = torch.from_numpy(text_feats[:, organ_idx, :]).to(device)
        sim = img @ txt.t()
        sim.fill_diagonal_(float("-inf"))  # exclude the patient's own report
        top_k_idx = sim.topk(top_k, dim=1).indices.cpu().numpy()
        organ_valid = image_valid[:, organ_idx]
        top_k_idx[~organ_valid] = -1
        all_indices[:, organ_idx, :] = top_k_idx
    return all_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text-embeddings", default=os.path.join(DATA_ROOT, "EK_files", "organ_report_embeddings.npz"),
        help="organ_report_embeddings.npz from compute_text_embeddings.py",
    )
    parser.add_argument(
        "--finetuned-checkpoint", default=_default_checkpoint(),
        help="a finetune.py checkpoint_XXX.pth with real trained weights for the 6 "
             "non-frozen organs - use a converged one, not an early one; defaults to "
             f"the highest-numbered one in {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--output-dir", default=os.path.join(DATA_ROOT, "EK_files"),
        help="where to write organ_annotation.json",
    )
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--max-samples", type=int, default=None, help="for smoke-testing on a subset")
    args = parser.parse_args()
    if args.finetuned_checkpoint is None:
        parser.error(f"no checkpoint found under {DEFAULT_OUTPUT_DIR} and --finetuned-checkpoint not given")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    npz = np.load(args.text_embeddings)
    text_feats = npz["feats"]
    patient_ids = [str(p) for p in npz["patient_ids"]]
    assert list(npz["organs"]) == ORGANS, "organ order in npz doesn't match current ORGANS"
    if args.max_samples is not None:
        patient_ids = patient_ids[:args.max_samples]
        text_feats = text_feats[:args.max_samples]
    print(f"{len(patient_ids)} patients, {len(ORGANS)} organs")
    print(f"using finetuned checkpoint: {args.finetuned_checkpoint}")

    captions = build_organ_captions(RADGENOME_CSV, ORGANS)

    model = build_eval_model(args.finetuned_checkpoint, device)

    image_feats, image_valid = compute_image_embeddings(model, patient_ids, device)
    print(f"image embeddings computed for {image_valid.sum()}/{image_valid.size} (patient, organ) pairs")

    top_k_indices = compute_top_k_indices(image_feats, image_valid, text_feats, args.top_k, device)

    records = []
    for i, pid in enumerate(patient_ids):
        record = {"id": pid, "image_path": f"{pid}.nii.gz"}
        for organ_idx, organ in enumerate(ORGANS):
            record[organ] = clean_caption(captions[pid][organ], organ)
            record[f"{organ}_indices"] = top_k_indices[i, organ_idx].tolist()
        records.append(record)
    assert len(records) == len(patient_ids), "record count doesn't match patient count"

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "organ_annotation.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"train": records}, f, ensure_ascii=False)
    print(f"Saved {json_path}: {len(records)} records")


if __name__ == "__main__":
    main()
