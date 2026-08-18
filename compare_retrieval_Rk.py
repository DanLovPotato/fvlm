"""Compare the original (un-finetuned) checkpoint vs an adapter-finetuned checkpoint
on per-organ image-text retrieval (R@1/R@5/R@10), using the held-out validation set
(processed_valid_images/masks + validation_region_report.csv - never touched by
finetune.py's training).

Not loss-based: this measures whether the model can actually retrieve the correct
report for a given organ image out of the whole validation corpus, per organ,
including the 6 organs the adapter added that the original checkpoint never saw.

IMPORTANT interpretation note for the "original" model on the 6 new organs: the
original checkpoint has no query_tokens/vision_proj for them at all (it only ever
had lung/heart/esophagus/aorta). So "original" here means the same expand_organs()
step build_eval_model() always does - frozen organs (lung/heart/esophagus) keep their
real pretrained weights, but the 6 new organs get freshly-initialized, UNTRAINED
heads, since that's genuinely all the original model has to offer for them. Orig R@k
on new organs is therefore expected to sit near chance level by construction - that's
the baseline the adapter's R@k is meant to beat, not a bug.

Retrieval semantics: standard image-to-text R@k, NOT find_similar_reports.py's
nearest-OTHER-patient search. Query image i's correct answer is text i (that same
patient's own report) - self-match is the target here, not excluded.
"""
import argparse
import csv
import os

import numpy as np
import torch
import torch.nn.functional as F
from monai import transforms
from tqdm import tqdm

from finetune import (
    CHECKPOINT_PATH, CROP_SIZE, DATA_ROOT, FROZEN_ORGANS, ORGANS, PATCH_SIZE,
    _apply_environment_patches, build_organ_captions, build_pretrained_model, expand_organs,
)
from eval_finetune import center_crop

VALID_IMAGE_ROOT = os.path.join(DATA_ROOT, "processed_valid_images")
VALID_MASK_ROOT = os.path.join(DATA_ROOT, "processed_valid_masks")
VALID_REGION_REPORT_CSV = os.path.join(DATA_ROOT, "radgenome_files", "validation_region_report.csv")

NEW_ORGANS = [o for o in ORGANS if o not in FROZEN_ORGANS]


def clean_caption(text, organ):
    """Matches BlipPretrain.forward()'s own caption-cleaning step exactly
    (blip_pretrain.py:220-224)."""
    template = f"{organ} shows no significant abnormalities."
    if text.startswith(template) and text != template:
        return text.replace(template, "")
    return text


def load_model(device, finetuned_checkpoint=None):
    """Base 4-organ checkpoint -> expand to 9 organs. Frozen organs keep their real
    pretrained weights either way. If finetuned_checkpoint is given, load it
    (strict=True) on top to get the adapter-trained model; otherwise leave the 6 new
    organs at their freshly-initialized, untrained expand_organs() state - see module
    docstring for why that's the only coherent "original" baseline for them.
    """
    _apply_environment_patches()
    model = build_pretrained_model(CHECKPOINT_PATH)
    expand_organs(model, ORGANS, FROZEN_ORGANS)
    if finetuned_checkpoint is not None:
        ckpt = torch.load(finetuned_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"], strict=True)
    return model.to(device).eval()


def find_valid_samples(captions, max_samples=None):
    """Mirrors CTOrganDataset's own file-filtering (image + mask + caption all
    present), just pointed at the held-out valid roots instead of the train ones."""
    samples = []
    for file_name in sorted(os.listdir(VALID_IMAGE_ROOT)):
        if not file_name.endswith(".nii.gz"):
            continue
        sample_id = file_name[:-len(".nii.gz")]
        label_path = os.path.join(VALID_MASK_ROOT, file_name)
        if not os.path.exists(label_path) or sample_id not in captions:
            continue
        samples.append((os.path.join(VALID_IMAGE_ROOT, file_name), label_path, sample_id))
    if max_samples is not None:
        samples = samples[:max_samples]
    return samples


@torch.inference_mode()
def embed_dataset(model, samples, captions, batch_size, device):
    """Text: tokenizer -> text_encoder -> text_proj, organ-agnostic (blip_pretrain.py:
    226-237). Image: per organ, center_crop -> ROI-pool -> attention -> vision_proj,
    the same forward_test_win path used everywhere else eval-time in this project
    (blip_pretrain.py:501-524) - one ViT forward per organ present, not per patient.
    Returns text_feats/image_feats (N, len(ORGANS), 256) float32 and image_valid
    (N, len(ORGANS)) bool (False where that organ's mask wasn't present at all).
    """
    patient_ids = [pid for _, _, pid in samples]
    n = len(samples)

    text_feats = np.zeros((n, len(ORGANS), 256), dtype=np.float32)
    for organ_idx, organ in enumerate(ORGANS):
        texts = [clean_caption(captions[pid][organ], organ) for pid in patient_ids]
        for start in tqdm(range(0, len(texts), batch_size), desc=f"text embed {organ}", leave=False):
            batch_texts = texts[start:start + batch_size]
            tokens = model.tokenizer(
                batch_texts, padding="max_length", truncation=True,
                max_length=model.max_txt_len, return_tensors="pt",
            ).to(device)
            text_output = model.text_encoder.forward_text(tokens)
            text_embeds = text_output.last_hidden_state
            text_feat = F.normalize(model.text_proj(text_embeds[:, 0, :]), dim=-1)
            text_feats[start:start + batch_size, organ_idx, :] = text_feat.cpu().numpy()

    loader = transforms.Compose([
        transforms.LoadImaged(keys=["image", "label"], image_only=True, ensure_channel_first=True),
    ])
    pad_func = transforms.DivisiblePadd(
        keys=["image", "label"], k=PATCH_SIZE, mode="constant", constant_values=0, method="end",
    )
    image_feats = np.zeros((n, len(ORGANS), 256), dtype=np.float32)
    image_valid = np.zeros((n, len(ORGANS)), dtype=bool)

    for i, (img_path, label_path, _pid) in enumerate(tqdm(samples, desc="image embed")):
        data = loader({"image": img_path, "label": label_path})
        image = data["image"].as_tensor()[None].to(device)
        mask = data["label"].as_tensor()[None].to(device)

        whole_organ_sizes = {
            organ: torch.eq(mask, ORGANS.index(organ) + 1).sum().item() for organ in ORGANS
        }
        organ_feat_dict = {}
        for organ_id, organ in enumerate(ORGANS):
            if whole_organ_sizes[organ] == 0:
                continue

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
                image_feats[i, organ_id, :] = np.array(organ_feat_dict[organ][0], dtype=np.float32)
                image_valid[i, organ_id] = True

    return text_feats, image_feats, image_valid


def compute_similarity(image_feats, text_feats, organ_idx):
    """(N, N) image-to-text cosine similarity for one organ - both sides are already
    L2-normalized, so a plain dot product is cosine similarity."""
    img = torch.from_numpy(image_feats[:, organ_idx, :])
    txt = torch.from_numpy(text_feats[:, organ_idx, :])
    return img @ txt.t()


def compute_recall_at_k(sim, valid_mask, k_list=(1, 5, 10, 20)):
    """Standard image-to-text retrieval R@k: for query image i, is the TRUE match
    (text i, the same patient's own report) within the top-k most similar texts?
    Self-match is the target here, not excluded (unlike find_similar_reports.py).
    Rows without a valid image embedding are dropped from the denominator, not
    scored as misses.
    """
    n = sim.shape[0]
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) == 0:
        return {k: float("nan") for k in k_list}, 0

    ranks = sim.argsort(dim=1, descending=True)
    results = {}
    for k in k_list:
        top_k = ranks[:, :k]
        hit = (top_k == torch.arange(n).unsqueeze(1)).any(dim=1).numpy()
        results[k] = hit[valid_idx].mean()
    return results, len(valid_idx)


def build_comparison_table(orig_text_feats, orig_image_feats, orig_valid,
                            adapt_text_feats, adapt_image_feats, adapt_valid):
    rows = []
    for organ_idx, organ in enumerate(ORGANS):
        orig_sim = compute_similarity(orig_image_feats, orig_text_feats, organ_idx)
        orig_recall, _ = compute_recall_at_k(orig_sim, orig_valid[:, organ_idx])

        adapt_sim = compute_similarity(adapt_image_feats, adapt_text_feats, organ_idx)
        adapt_recall, adapt_n = compute_recall_at_k(adapt_sim, adapt_valid[:, organ_idx])

        rows.append({
            "organ": organ,
            "is_new": organ in NEW_ORGANS,
            "n": adapt_n,
            "orig_r1": orig_recall[1], "orig_r5": orig_recall[5], "orig_r10": orig_recall[10], "orig_r20": orig_recall[20],
            "adapt_r1": adapt_recall[1], "adapt_r5": adapt_recall[5], "adapt_r10": adapt_recall[10], "adapt_r20": adapt_recall[20],
            "delta_r1": adapt_recall[1] - orig_recall[1],
        })
    return rows


def print_and_save_table(rows, output_dir):
    header = ["Organ", "N", "Orig R@1", "Orig R@5", "Orig R@10", "Orig R@20",
              "Adapter R@1", "Adapter R@5", "Adapter R@10", "Adapter R@20", "Delta R@1"]
    lines_md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines_csv = [header]

    for r in rows:
        tag = " (new)" if r["is_new"] else ""
        vals = [
            f"{r['organ']}{tag}", str(r["n"]),
            f"{r['orig_r1']:.3f}", f"{r['orig_r5']:.3f}", f"{r['orig_r10']:.3f}", f"{r['orig_r20']:.3f}",
            f"{r['adapt_r1']:.3f}", f"{r['adapt_r5']:.3f}", f"{r['adapt_r10']:.3f}", f"{r['adapt_r20']:.3f}",
            f"{r['delta_r1']:+.3f}",
        ]
        lines_md.append("| " + " | ".join(vals) + " |")
        lines_csv.append(vals)

    new_rows = [r for r in rows if r["is_new"]]
    summary = {
        "orig_r1": float(np.mean([r["orig_r1"] for r in new_rows])),
        "orig_r5": float(np.mean([r["orig_r5"] for r in new_rows])),
        "orig_r10": float(np.mean([r["orig_r10"] for r in new_rows])),
        "orig_r20": float(np.mean([r["orig_r20"] for r in new_rows])),
        "adapt_r1": float(np.mean([r["adapt_r1"] for r in new_rows])),
        "adapt_r5": float(np.mean([r["adapt_r5"] for r in new_rows])),
        "adapt_r10": float(np.mean([r["adapt_r10"] for r in new_rows])),
        "adapt_r20": float(np.mean([r["adapt_r20"] for r in new_rows])),
    }
    summary_line = (
        f"\nNEW-ORGAN SUMMARY (mean over {[r['organ'] for r in new_rows]}):\n"
        f"  Original: R@1={summary['orig_r1']:.3f} R@5={summary['orig_r5']:.3f} R@10={summary['orig_r10']:.3f} R@20={summary['orig_r20']:.3f}\n"
        f"  Adapter:  R@1={summary['adapt_r1']:.3f} R@5={summary['adapt_r5']:.3f} R@10={summary['adapt_r10']:.3f} R@20={summary['adapt_r20']:.3f}\n"
        f"  Delta R@1: {summary['adapt_r1'] - summary['orig_r1']:+.3f}"
    )

    table_text = "\n".join(lines_md) + "\n" + summary_line + "\n"
    print(table_text)

    os.makedirs(output_dir, exist_ok=True)
    md_path = os.path.join(output_dir, "retrieval_comparison.md")
    with open(md_path, "w") as f:
        f.write(table_text)

    csv_path = os.path.join(output_dir, "retrieval_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(lines_csv)

    print(f"Saved {md_path}")
    print(f"Saved {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetuned-checkpoint", required=True)
    parser.add_argument("--output-dir", default=os.path.join(DATA_ROOT, "retrieval_comparison"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=None, help="for smoke-testing on a subset")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    captions = build_organ_captions(VALID_REGION_REPORT_CSV, ORGANS)
    samples = find_valid_samples(captions, args.max_samples)
    print(f"{len(samples)} validation patients, {len(ORGANS)} organs ({len(NEW_ORGANS)} new: {NEW_ORGANS})")

    print("\n=== embedding with ORIGINAL (un-finetuned) model ===")
    orig_model = load_model(device, finetuned_checkpoint=None)
    orig_text_feats, orig_image_feats, orig_valid = embed_dataset(
        orig_model, samples, captions, args.batch_size, device)
    del orig_model
    torch.cuda.empty_cache()

    print("\n=== embedding with ADAPTER (finetuned) model ===")
    adapt_model = load_model(device, finetuned_checkpoint=args.finetuned_checkpoint)
    adapt_text_feats, adapt_image_feats, adapt_valid = embed_dataset(
        adapt_model, samples, captions, args.batch_size, device)
    del adapt_model
    torch.cuda.empty_cache()

    rows = build_comparison_table(
        orig_text_feats, orig_image_feats, orig_valid,
        adapt_text_feats, adapt_image_feats, adapt_valid,
    )
    print_and_save_table(rows, args.output_dir)


if __name__ == "__main__":
    main()
