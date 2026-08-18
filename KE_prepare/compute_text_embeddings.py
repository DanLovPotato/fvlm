"""Step 1 of 2 for the MvKeTR-style knowledge base: compute the per-organ TEXT
embedding corpus (organ_report_embeddings.npz).

Split out from the retrieval step (find_similar_reports.py) because this half is
fine-tuning-independent and can run right now: text_encoder/text_proj are frozen
throughout finetune.py's whole run (freeze_shared_modules), so these embeddings are
identical whether finetune.py is 1 epoch in or fully converged - no need to wait for
or redo this once fine-tuning finishes. The retrieval step, by contrast, needs each
patient's own IMAGE embedding (via the model's finetuned vision_projs for the 6
non-frozen organs), which only means anything once fine-tuning has actually
converged - see find_similar_reports.py.

Output: organ_report_embeddings.npz - feats (N, len(ORGANS), 256) float32,
L2-normalized text embeddings; patient_ids (N,) str; organs (len(ORGANS),) str,
matching ORGANS' order.
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finetune import (
    CHECKPOINT_PATH, CTOrganDataset, DATA_ROOT, ORGANS, RADGENOME_CSV,
    _apply_environment_patches, build_organ_captions, build_pretrained_model,
)


def clean_caption(text, organ):
    """Matches BlipPretrain.forward()'s own caption-cleaning step exactly
    (blip_pretrain.py:220-224), so these embeddings match what training/eval computes
    for the same caption text."""
    template = f"{organ} shows no significant abnormalities."
    if text.startswith(template) and text != template:
        return text.replace(template, "")
    return text


@torch.inference_mode()
def compute_text_embeddings(model, patient_ids, captions, batch_size, device):
    feats = np.zeros((len(patient_ids), len(ORGANS), 256), dtype=np.float32)
    for organ_idx, organ in enumerate(ORGANS):
        texts = [clean_caption(captions[pid][organ], organ) for pid in patient_ids]
        for start in tqdm(range(0, len(texts), batch_size), desc=f"embed {organ}"):
            batch_texts = texts[start:start + batch_size]
            tokens = model.tokenizer(
                batch_texts, padding="max_length", truncation=True,
                max_length=model.max_txt_len, return_tensors="pt",
            ).to(device)
            text_output = model.text_encoder.forward_text(tokens)
            text_embeds = text_output.last_hidden_state
            text_feat = F.normalize(model.text_proj(text_embeds[:, 0, :]), dim=-1)
            feats[start:start + batch_size, organ_idx, :] = text_feat.cpu().numpy()
    return feats


def main():
    parser = argparse.ArgumentParser()
    # Native 4-organ CHECKPOINT_PATH, not a finetuned checkpoint - text_encoder/
    # text_proj are identical between the released and any finetuned checkpoint
    # (frozen), so expand_organs()/a finetuned checkpoint are irrelevant here.
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    parser.add_argument(
        "--output-dir", default=os.path.join(DATA_ROOT, "EK_files"),
        help="where to write organ_report_embeddings.npz",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=None, help="for smoke-testing on a subset")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    captions = build_organ_captions(RADGENOME_CSV, ORGANS)
    # Reuses CTOrganDataset's own file-filtering (image + mask + caption all present)
    # so the patient list here matches exactly what finetune.py itself trains on, and
    # what find_similar_reports.py will later need image embeddings for -
    # vis_processor is only touched by __getitem__, which we never call.
    dataset = CTOrganDataset(ORGANS, captions, vis_processor=None)
    samples = dataset.samples
    if args.max_samples is not None:
        samples = samples[:args.max_samples]
    patient_ids = [sample_id for _, _, sample_id in samples]
    print(f"{len(patient_ids)} patients, {len(ORGANS)} organs")

    _apply_environment_patches()
    model = build_pretrained_model(args.checkpoint).to(device).eval()

    feats = compute_text_embeddings(model, patient_ids, captions, args.batch_size, device)
    assert not np.isnan(feats).any(), "NaN in computed text embeddings"

    os.makedirs(args.output_dir, exist_ok=True)
    npz_path = os.path.join(args.output_dir, "organ_report_embeddings.npz")
    np.savez(npz_path, feats=feats, patient_ids=np.array(patient_ids), organs=np.array(ORGANS))
    print(f"Saved {npz_path}: feats shape {feats.shape}")

    print("\nper-organ non-template (real finding) sample counts:")
    for organ in ORGANS:
        template = f"{organ} shows no significant abnormalities."
        n_real = sum(1 for pid in patient_ids if captions[pid][organ] != template)
        print(f"  {organ}: {n_real}/{len(patient_ids)}")


if __name__ == "__main__":
    main()
