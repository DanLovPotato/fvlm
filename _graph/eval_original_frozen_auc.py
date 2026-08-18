"""Sanity check: does eval_finetune.py's AUC scoring for the frozen organs
(lung/heart/esophagus) match what you'd get from the untouched, released checkpoint in
its own NATIVE 4-organ structure (self.organs = ['lung','heart','esophagus','aorta'],
mask ids 1..4) - i.e. no expand_organs() call at all, not even to remap organ ids.

This project's mask files use preprocess.py's 9-organ numbering instead (abdomen=1,
bone=2, breast=3, esophagus=4, heart=5, lung=6, mediastinum=7, thyroid=8, trachea and
bronchie=9) - completely different ids than the native model expects. Feeding those
mask values straight into the native model would silently score the wrong organ (see
eval_finetune.py's module docstring, point 1) or, for the higher-numbered organs,
IndexError against the native model's 4-entry organs list. So instead of touching the
model at all, this script remaps the mask values for just lung/heart/esophagus onto
the native scheme (1/2/3) and zeroes out every other voxel, then runs the untouched
native model against that.

Reuses eval_finetune.py's TEST_ITEMS/PATHOLOGY_KEYWORDS/scoring machinery directly so
the numbers are directly comparable to the per-epoch AUCs already collected under
eval_finetune_logs/ - if expand_organs() really does preserve frozen-organ weights
untouched (as its docstring in finetune.py claims), these numbers should match those
exactly.
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
from monai import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# Lives in _graph/ - add the repo root so finetune.py/eval_finetune.py are importable
# regardless of cwd, matching plot_eval_finetune_trends.py's approach.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import eval_finetune
from finetune import (
    CHECKPOINT_PATH, CROP_SIZE, DATA_ROOT, FROZEN_ORGANS, ORGANS, PATCH_SIZE,
    _apply_environment_patches, build_pretrained_model,
)
from eval_finetune import (
    EvalOrganDataset, TEST_ITEMS, collate_fn, center_crop, load_ground_truth,
    score_against_ground_truth, vqa_abnormality_csv,
)

# The checkpoint's own native organ list/id scheme (BlipPretrain.__init__,
# blip_pretrain.py:105) - untouched, not expand_organs()'s remapped version.
NATIVE_ORGANS = ["lung", "heart", "esophagus", "aorta"]
FROZEN_TEST_ITEMS = [item for item in TEST_ITEMS if item[0] in FROZEN_ORGANS]


def remap_mask_to_native(mask):
    """Project's 9-organ mask ids -> native model's 4-organ ids, for the 3 organs
    that exist in both. Every other voxel (including aorta, which has no mask here)
    becomes 0."""
    native_mask = torch.zeros_like(mask)
    for organ in FROZEN_ORGANS:
        project_id = ORGANS.index(organ) + 1
        native_id = NATIVE_ORGANS.index(organ) + 1
        native_mask[mask == project_id] = native_id
    return native_mask


@torch.inference_mode()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _apply_environment_patches()
    model = build_pretrained_model(CHECKPOINT_PATH)  # native shape, no expand_organs
    model = model.to(device).eval()
    assert model.organs == NATIVE_ORGANS, f"unexpected native organ order: {model.organs}"

    image_root = f"{DATA_ROOT}/processed_valid_images"
    mask_root = f"{DATA_ROOT}/processed_valid_masks"
    dataset = EvalOrganDataset(ORGANS, FROZEN_TEST_ITEMS, image_root=image_root, mask_root=mask_root)
    print(f"Evaluating NATIVE (un-expanded) checkpoint on {len(dataset)} samples, "
          f"frozen organs only: {sorted(FROZEN_ORGANS)}, model.organs={model.organs}")

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4,
                         drop_last=False, collate_fn=collate_fn)
    pad_func = transforms.DivisiblePadd(
        keys=["image", "label"], k=PATCH_SIZE, mode="constant", constant_values=0, method="end",
    )

    text_feat_dict = model.prepare_text_feat(FROZEN_TEST_ITEMS)
    organ_feat_dict = {}
    results = []

    for image, mask, test_items, meta_info in tqdm(loader, desc="Infer"):
        fid = meta_info["file_name"]
        organ_feat_dict[fid] = {}
        image = image[None].to(device)
        mask = remap_mask_to_native(mask[None].to(device))

        test_organs = [o for o in meta_info["test_organ_names"] if o in FROZEN_ORGANS]
        whole_organ_sizes = dict(zip(
            test_organs, [torch.eq(mask, NATIVE_ORGANS.index(o) + 1).sum().item() for o in test_organs],
        ))
        test_organs = [o for o in test_organs if whole_organ_sizes[o] > 0]
        test_items = [item for item in test_items if item[0] in test_organs]

        organ_logits = dict(zip(test_items, [[] for _ in test_items]))
        for k, v in list(organ_logits.items()):
            if len(v):
                continue
            organ_name = k[0]
            organ_id = NATIVE_ORGANS.index(organ_name)
            window_patch, window_mask = center_crop(image, torch.eq(mask, organ_id + 1), crop_size=CROP_SIZE)
            window_mask = window_mask.float()
            window_mask[window_mask == 1] = organ_id + 1
            pad_data = pad_func({"image": window_patch[0], "label": window_mask[0]})
            window_patch, window_mask = pad_data["image"], pad_data["label"]
            organ_logits = model.forward_test_win(
                window_patch[None], window_mask[None], organ_logits, test_organs,
                text_feat_dict, organ_feat_dict[fid], whole_organ_sizes, skip_organ=organ_id,
            )

        row = [fid] + [""] * len(FROZEN_TEST_ITEMS)
        organ_logits = {item: probs for item, probs in organ_logits.items() if len(probs) > 0}
        for item, probs in organ_logits.items():
            row[FROZEN_TEST_ITEMS.index(item) + 1] = np.concatenate(probs).mean(0)[1]
        results.append(row)

    df = pd.DataFrame(results, columns=["file_name"] + ["_".join(item[:2]) for item in FROZEN_TEST_ITEMS])

    vqa_csv = vqa_abnormality_csv("valid")
    ground_truth = load_ground_truth(vqa_csv)
    # score_against_ground_truth() reads the module-level TEST_ITEMS global inside
    # eval_finetune.py - point it at just the frozen subset so it doesn't KeyError on
    # the trainable-organ columns we never computed here.
    eval_finetune.TEST_ITEMS = FROZEN_TEST_ITEMS
    score_against_ground_truth(df, ground_truth)


if __name__ == "__main__":
    main()
