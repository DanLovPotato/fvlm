"""Zero-shot pathology-classification eval for the finetuned 9-organ checkpoint,
adapted from fvlm_original/eval.py - kept as a separate script (not modifying
eval.py) since eval.py is upstream reference code built for the original 4-organ
release checkpoint, the same way finetune.py itself was adapted from the original
training script without editing it.

Three things differ from eval.py, matching the same kind of adaptations finetune.py
made to the original training script:

1. Organ-id mapping: eval.py assumes organs=['lung','heart','esophagus','aorta'] with
   ids 1..4. Our merged masks use the 9-organ order from preprocess.py's ORGANS list
   (lung=6, no aorta) - using eval.py's id scheme against our mask files would silently
   score the wrong organ's voxels, not crash. TEST_ITEMS below drops the 'aorta' item
   (no aorta mask exists in this project) and every organ-id lookup goes through
   ORGANS.index(...) instead of a hardcoded 4-organ list.

2. Model construction: eval.py builds the model at its native 4-organ shape via
   model_cls.from_config() and loads a checkpoint with strict=False. Our finetuned
   checkpoint's query_tokens/vision_projs are shaped for 9 organs -
   load_state_dict tolerates missing/extra *keys* under strict=False but still raises
   on a shape mismatch for a key present in both, so this would crash. build_eval_model()
   below reuses finetune.py's own build_pretrained_model()+expand_organs() two-step
   (imported directly, not copied - finetune.py is this project's own code, unlike the
   vendored eval.py/blip_pretrain.py) to get the same 9-organ skeleton finetune.py
   trained, then loads our finetuned state dict into it with strict=True.

3. Data layout: eval.py's DataFolder hardcodes vis_root='data/processed_valid_images'
   (relative, doesn't exist here) and does mask_path = image_path.replace('images',
   'masks'). EvalOrganDataset below defaults to this project's PREPROCESSED_IMAGE_ROOT/
   PREPROCESSED_MASK_ROOT (the same constants finetune.py's CTOrganDataset uses) but
   takes image_root/mask_root overrides - --split valid (the default; see parse_args())
   points it at processed_valid_images/masks instead, a genuinely held-out patient set
   finetune.py's training never touches, unlike the train folders. Image/mask pairing
   is by matching filename across the two directories, like CTOrganDataset does.

masks_to_boxes_3d() and center_crop() are copied verbatim from eval.py (pure tensor
math, no organ-count or path assumptions) rather than imported, since eval.py is
being kept untouched as upstream reference here rather than treated as an importable
dependency of this project's own scripts. eval.py's unused sliding-window setup
(_get_scan_interval/dense_patch_slices/num_win - computed but never read before the
loop falls through to a single center_crop per organ) and its distributed-eval
plumbing aren't carried over, matching finetune.py's own single-GPU-only style.

Historical note, now resolved: ORGANS in finetune.py/preprocess.py used to include
'pleura' (its raw mask was voxel-for-voxel identical to lung's, which silently erased
lung's label every time merge_organ_masks() ran - see finetune.py's module
docstring). 'pleura' was removed from ORGANS to fix that, which shifted every organ
listed after it (old thyroid=9/trachea=10 -> new thyroid=8/trachea=9). Both
processed_valid_masks/images and every checkpoint this script can use are now
regenerated/trained under the current 9-organ numbering, so this no longer needs
checking before running.

Only lung/heart/esophagus have curated pathology text prompts here (carried over from
eval.py's original CT-RATE-derived list). The 6 organs added by finetune.py's
expand_organs() (abdomen, bone, breast, mediastinum, thyroid, trachea and bronchie)
have partial coverage below, added after checking keyword support against
validation_vqa_abnormality.csv (see PATHOLOGY_KEYWORDS).
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from monai import transforms
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from finetune import (
    CHECKPOINT_PATH, CROP_SIZE, DATA_ROOT, DEFAULT_OUTPUT_DIR, FROZEN_ORGANS, ORGANS,
    PATCH_SIZE, PREPROCESSED_IMAGE_ROOT, PREPROCESSED_MASK_ROOT,
    _apply_environment_patches, build_pretrained_model, expand_organs,
)

# Ground truth for scoring TEST_ITEMS predictions - free-text abnormality findings per
# (patient, organ), NOT a binary structured-label file (no such file exists in this
# project's data; see the conversation this was added in for what was checked).
def vqa_abnormality_csv(split):
    """train/valid ground-truth files are keyed by disjoint patient-id prefixes
    (train_XXX vs valid_XXX) - must match whichever split EvalOrganDataset is
    actually reading images from, or every row silently fails to match. Naming is
    inconsistent between the data folders (train_/valid_, short form) and these CSVs
    (train_/validation_, full word) - not something to "fix" file-side, just map it.
    """
    csv_prefix = {"train": "train", "valid": "validation"}[split]
    return os.path.join(DATA_ROOT, "radgenome_files", f"{csv_prefix}_vqa_abnormality.csv")

# eval.py's original 4-organ test_items (minus 'aorta' - no aorta mask exists in this
# project's organ scheme, see module docstring point 1) extended to all organs ORGANS
# now covers. 'pleura' isn't one of them - dropped from ORGANS itself in finetune.py/
# preprocess.py because its raw mask was voxel-for-voxel identical to lung's, which was
# silently erasing lung's label during merge_organ_masks() (see finetune.py's module
# docstring for the full explanation). Every extension item below was only added after
# checking its keyword actually appears at least 15 times in real free text - first
# pass against validation_vqa_abnormality.csv (1,564 patients), later cross-checked
# against the much larger train_vqa_abnormality.csv (24,123 patients) once that became
# available, which both confirmed the original picks and surfaced several more
# (abdomen/bone additions below) that weren't obviously frequent at the smaller sample
# size. Candidates that came back at 0 or single digits even in the larger file (e.g.
# mediastinum 'aortic aneurysm', thyroid 'goiter', breast 'breast mass') were left out.
TEST_ITEMS = [
    ('lung', 'Emphysema', 'Not Emphysema.', 'Emphysema.'),
    ('lung', 'Atelectasis', 'Not Atelectatic.', 'Atelectatic.'),
    ('lung', 'Lung nodule', 'Not Nodule.', 'Nodule.'),
    ('lung', 'Lung opacity', 'Not Opacity.', 'Opacity.'),
    ('lung', 'Pulmonary fibrotic sequela', 'Not Pulmonary fibrotic.', 'Pulmonary fibrotic.'),
    ('lung', 'Pleural effusion', 'Not Pleural effusion.', 'Pleural effusion.'),
    ('lung', 'Mosaic attenuation pattern', 'Not Mosaic attenuation pattern.', 'Mosaic attenuation pattern.'),
    ('lung', 'Peribronchial thickening', 'Not Peribronchial thickening.', 'Peribronchial thickening.'),
    ('lung', 'Consolidation', 'Not Consolidation.', 'Consolidation.'),
    ('lung', 'Bronchiectasis', 'Not Bronchiectasis.', 'Bronchiectasis.'),
    ('lung', 'Interlobular septal thickening', 'Not Interlobular septal thickening.', 'Interlobular septal thickening.'),
    ('lung', 'Pneumonia / COVID-19 pattern', 'Not Pneumonia.', 'Pneumonia.'),
    ('heart', 'Cardiomegaly', 'Not Cardiomegaly.', 'Cardiomegaly.'),
    ('heart', 'Pericardial effusion', 'Not Pericardial effusion.', 'Pericardial effusion.'),
    ('heart', 'Coronary artery wall calcification', 'Not Coronary artery wall calcification.', 'Coronary artery wall calcification.'),
    ('esophagus', 'Hiatal hernia', 'Not Hiatal hernia.', 'Hiatal hernia.'),
    ('mediastinum', 'Arterial wall calcification', 'Not Arterial wall calcification.', 'Arterial wall calcification.'),
    ('mediastinum', 'Aortic dilation', 'Not Aortic dilation.', 'Aortic dilation.'),
    ('mediastinum', 'Lymphadenopathy', 'Not Lymphadenopathy.', 'Lymphadenopathy.'),
    ('bone', 'Degenerative changes', 'Not Degenerative changes.', 'Degenerative changes.'),
    ('bone', 'Scoliosis', 'Not Scoliosis.', 'Scoliosis.'),
    ('bone', 'Osteopenia', 'Not Osteopenia.', 'Osteopenia.'),
    ('bone', 'Osteoporosis', 'Not Osteoporosis.', 'Osteoporosis.'),
    ('bone', 'Kyphosis', 'Not Kyphosis.', 'Kyphosis.'),
    ('bone', 'Vertebral height loss', 'Not Vertebral height loss.', 'Vertebral height loss.'),
    ('thyroid', 'Thyroid nodule', 'Not Thyroid nodule.', 'Thyroid nodule.'),
    ('abdomen', 'Free abdominal fluid', 'Not Free abdominal fluid.', 'Free abdominal fluid.'),
    ('abdomen', 'Arterial wall calcification', 'Not Arterial wall calcification.', 'Arterial wall calcification.'),
    ('abdomen', 'Hepatic steatosis', 'Not Hepatic steatosis.', 'Hepatic steatosis.'),
    ('abdomen', 'Renal cortical cyst', 'Not Renal cortical cyst.', 'Renal cortical cyst.'),
    ('abdomen', 'Calculus', 'Not Calculus.', 'Calculus.'),
    ('abdomen', 'Adenoma', 'Not Adenoma.', 'Adenoma.'),
    ('abdomen', 'Hypodense lesion', 'Not Hypodense lesion.', 'Hypodense lesion.'),
    ('breast', 'Gynecomastia', 'Not Gynecomastia.', 'Gynecomastia.'),
    ('trachea and bronchie', 'Peribronchial thickening', 'Not Peribronchial thickening.', 'Peribronchial thickening.'),
    ('trachea and bronchie', 'Bronchiectasis', 'Not Bronchiectasis.', 'Bronchiectasis.'),
    ('trachea and bronchie', 'Tracheal diverticulum', 'Not Tracheal diverticulum.', 'Tracheal diverticulum.'),
]

# Keyword(s) that must appear (after lowercasing and collapsing '-' to ' ') in a
# patient's top-level Abnormality text for that organ, for a TEST_ITEMS entry to count
# as a ground-truth positive. Curated by directly checking the real free-text phrasing
# in validation_vqa_abnormality.csv (and cross-checked against the larger
# train_vqa_abnormality.csv once available), NOT CT-RATE's structured pathology names,
# which mostly don't appear verbatim (e.g. "lung opacity" almost never appears
# literally - it shows up as "ground glass opacities" instead). Items not listed here
# have no ground-truth check - only a raw confidence score, same as before this was
# added.
# 'coronary artery wall calcification' is deliberately narrow (just 'coronary') rather
# than the broader 'calcific'/'atheroscl' terms that also show up for non-coronary
# (e.g. valve, pericardial) calcification on heart-level rows - conservative
# under-matching was chosen over conflating different calcification sites. Similarly,
# abdomen's 'Hypodense lesion' is intentionally organ-nonspecific (could be liver,
# kidney, spleen, etc - the phrase alone doesn't say) - included anyway since the
# finding itself (943+ mentions) is real and common, just less anatomically precise
# than the other abdomen items.
# 'cardiomegaly'/'aortic dilation' both lean on measurement-language synonyms ('wider
# than normal', 'cardiothoracic index/ratio') rather than just the diagnostic word
# itself - reports here describe heart/vessel size that way far more often than saying
# "cardiomegaly"/"dilation" outright (10 vs 69 mentions for cardiomegaly alone vs with
# synonyms, checked against train_vqa_abnormality.csv).
# A 'cto' (coronary chronic total occlusion) keyword was considered and dropped:
# substring-matching it picks up "pneumonectomy" (contains "...ne-cto-my") as a false
# positive, which exact-phrase frequency counting alone didn't reveal.
PATHOLOGY_KEYWORDS = {
    ('lung', 'Emphysema'): ['emphysem'],
    ('lung', 'Atelectasis'): ['atelecta'],
    ('lung', 'Lung nodule'): ['nodul'],
    ('lung', 'Lung opacity'): ['opacit', 'ground glass'],
    ('lung', 'Pulmonary fibrotic sequela'): ['fibro'],
    ('lung', 'Pleural effusion'): ['pleural effusion'],
    ('lung', 'Consolidation'): ['consolidat'],
    ('lung', 'Bronchiectasis'): ['bronchiecta'],
    ('lung', 'Interlobular septal thickening'): ['septal thick', 'interlobular'],
    ('lung', 'Peribronchial thickening'): ['peribronchial'],
    ('lung', 'Mosaic attenuation pattern'): ['mosaic'],
    # 'pneumon' (not 'pneumonia') so it also catches "pneumonic infiltration"/
    # "viral pneumonias", which don't contain the literal substring 'pneumonia'.
    ('lung', 'Pneumonia / COVID-19 pattern'): ['pneumon', 'covid'],
    ('heart', 'Cardiomegaly'): ['cardiomegaly', 'wider than normal', 'larger than normal', 'cardiothoracic'],
    ('heart', 'Pericardial effusion'): ['pericardial effusion'],
    ('heart', 'Coronary artery wall calcification'): ['coronary'],
    ('esophagus', 'Hiatal hernia'): ['hiatal hernia'],
    ('mediastinum', 'Arterial wall calcification'): ['calcific', 'atheroscl'],
    ('mediastinum', 'Aortic dilation'): ['dilat', 'wider than normal', 'above normal'],
    ('mediastinum', 'Lymphadenopathy'): ['lymph node'],
    ('bone', 'Degenerative changes'): ['degenerative'],
    ('bone', 'Scoliosis'): ['scoliosis'],
    ('bone', 'Osteopenia'): ['osteopenia'],
    ('bone', 'Osteoporosis'): ['osteoporosis'],
    ('bone', 'Kyphosis'): ['kyphosis'],
    ('bone', 'Vertebral height loss'): ['height loss'],
    ('thyroid', 'Thyroid nodule'): ['nodul'],
    ('abdomen', 'Free abdominal fluid'): ['free fluid'],
    ('abdomen', 'Arterial wall calcification'): ['calcific', 'atheroscl', 'atheroma'],
    ('abdomen', 'Hepatic steatosis'): ['steatosis', 'hepatosteat'],
    ('abdomen', 'Renal cortical cyst'): ['cortical cyst'],
    ('abdomen', 'Calculus'): ['calculus', 'stones'],
    ('abdomen', 'Adenoma'): ['adenoma'],
    ('abdomen', 'Hypodense lesion'): ['hypodense lesion'],
    ('breast', 'Gynecomastia'): ['gynecomastia'],
    ('trachea and bronchie', 'Peribronchial thickening'): ['peribronchial'],
    ('trachea and bronchie', 'Bronchiectasis'): ['bronchiecta'],
    ('trachea and bronchie', 'Tracheal diverticulum'): ['diverticulum'],
}


def masks_to_boxes_3d(masks):
    """Copied from eval.py verbatim - see module docstring for why this is a copy,
    not an import."""
    if masks.numel() == 0:
        return torch.zeros((0, 6), device=masks.device)

    d, h, w = masks.shape[-3:]

    z = torch.arange(0, d, dtype=torch.float, device=masks.device)
    y = torch.arange(0, h, dtype=torch.float, device=masks.device)
    x = torch.arange(0, w, dtype=torch.float, device=masks.device)

    z, y, x = torch.meshgrid(z, y, x, indexing='ij')

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1).values
    x_min = x_mask.masked_fill(~masks.bool(), float('inf')).flatten(1).min(-1).values

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1).values
    y_min = y_mask.masked_fill(~masks.bool(), float('inf')).flatten(1).min(-1).values

    z_mask = (masks * z.unsqueeze(0))
    z_max = z_mask.flatten(1).max(-1).values
    z_min = z_mask.masked_fill(~masks.bool(), float('inf')).flatten(1).min(-1).values

    return torch.stack([x_min, y_min, z_min, x_max, y_max, z_max], dim=1)


def center_crop(image, mask, crop_size):
    """Copied from eval.py verbatim."""
    x_min, y_min, z_min, x_max, y_max, z_max = masks_to_boxes_3d(mask)[0].long()

    crop_d = max(crop_size[0], z_max - z_min)
    crop_h = max(crop_size[1], y_max - y_min)
    crop_w = max(crop_size[2], x_max - x_min)

    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2
    cz = (z_min + z_max) // 2

    d, h, w = image.shape[-3:]

    x_start = max(0, cx - crop_w // 2)
    x_end = min(w, x_start + crop_w)
    if x_end - x_start < crop_w:
        x_start = max(0, x_end - crop_w)

    y_start = max(0, cy - crop_h // 2)
    y_end = min(h, y_start + crop_h)
    if y_end - y_start < crop_h:
        y_start = max(0, y_end - crop_h)

    z_start = max(0, cz - crop_d // 2)
    z_end = min(d, z_start + crop_d)
    if z_end - z_start < crop_d:
        z_start = max(0, z_end - crop_d)

    return (
        image[..., z_start:z_end, y_start:y_end, x_start:x_end],
        mask[..., z_start:z_end, y_start:y_end, x_start:x_end],
    )


def collate_fn(batch):
    return batch[0]


class EvalOrganDataset(Dataset):
    """Mirrors eval.py's DataFolder, pointed at this project's data layout instead of
    eval.py's hardcoded 'data/processed_valid_images' + images->masks string
    replacement (see module docstring, point 3).

    image_root/mask_root default to PREPROCESSED_IMAGE_ROOT/PREPROCESSED_MASK_ROOT
    (the train folders) for backward compatibility, but --split valid in
    parse_args() below points these at the real held-out processed_valid_images/masks
    instead - the train folders were never a genuine held-out set.
    """

    def __init__(self, organs, test_items, image_root=None, mask_root=None):
        self.organs = organs
        self.test_items = test_items
        self.image_root = image_root or PREPROCESSED_IMAGE_ROOT
        self.mask_root = mask_root or PREPROCESSED_MASK_ROOT
        self.img_paths = [
            os.path.join(self.image_root, file_name)
            for file_name in sorted(os.listdir(self.image_root))
            if file_name.endswith(".nii.gz")
        ]
        self.loader = transforms.Compose([
            transforms.LoadImaged(keys=["image", "label"], image_only=True, ensure_channel_first=True),
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        image_path = self.img_paths[index]
        file_name = os.path.basename(image_path)
        mask_path = os.path.join(self.mask_root, file_name)

        data = self.loader({"image": image_path, "label": mask_path})
        meta_info = {
            "file_name": file_name,
            "img_path": image_path,
            "test_organ_names": self.organs,
        }
        return data["image"].as_tensor(), data["label"].as_tensor(), self.test_items, meta_info


def _default_checkpoint():
    if not os.path.isdir(DEFAULT_OUTPUT_DIR):
        return None
    candidates = sorted(
        f for f in os.listdir(DEFAULT_OUTPUT_DIR)
        if f.startswith("checkpoint_") and f.endswith(".pth")
    )
    return os.path.join(DEFAULT_OUTPUT_DIR, candidates[-1]) if candidates else None


def build_eval_model(finetuned_checkpoint_path, device):
    """Builds the same 10-organ skeleton finetune.py trains (native 4-organ
    BlipPretrain -> expand_organs to 10), then loads our finetuned weights into it
    with strict=True - not eval.py's model_cls.from_config() + strict=False, which
    would build at the wrong (4-organ) shape for this checkpoint. See module
    docstring, point 2.

    finetuned_checkpoint_path=None skips the load entirely, leaving the "before any
    finetuning" state expand_organs() itself produces: FROZEN_ORGANS keep their real
    pretrained weights, the other organs get freshly-initialized, untrained heads.
    This is the baseline to compare finetuned-epoch AUCs against - genuinely
    "no finetuning at all", not checkpoint_000.pth (which already reflects one
    training epoch).
    """
    _apply_environment_patches()
    model = build_pretrained_model(CHECKPOINT_PATH)
    expand_organs(model, ORGANS, FROZEN_ORGANS)

    if finetuned_checkpoint_path is not None:
        ckpt = torch.load(finetuned_checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"], strict=True)

    model = model.to(device)
    model.eval()
    return model


def load_ground_truth(vqa_csv_path):
    """(patient_id, organ) -> lowercased top-level Abnormality text, from
    validation_vqa_abnormality.csv. Only the exact top-level Anatomy==organ row is
    kept per patient - same principle finetune.py's build_organ_captions() uses:
    sub-anatomy rows (e.g. 'heart/heart tissue') are the same clinical content split
    finer, not additional information.
    """
    df = pd.read_csv(vqa_csv_path)
    df = df.dropna(subset=["Anatomy", "Abnormality"])
    df["patient_id"] = df["Volumename"].str.replace(".nii.gz", "", regex=False)

    gt = {}
    for row in df.itertuples():
        if row.Anatomy not in ORGANS:
            continue
        gt.setdefault(row.patient_id, {})[row.Anatomy] = str(row.Abnormality).strip().lower()
    return gt


def score_against_ground_truth(df, ground_truth):
    """For every TEST_ITEMS entry with a PATHOLOGY_KEYWORDS mapping, compares the
    model's predicted probability (>0.5 => positive) against a keyword-derived binary
    ground truth. This ground truth is a text-matching approximation of the radiology
    report, not a verified clinical label - read accuracy/AUC here as "does the
    model's confidence track the report's wording", not certified diagnostic accuracy.
    """
    print()
    print("accuracy against validation_vqa_abnormality.csv (keyword-derived ground "
          "truth, not verified clinical labels):")
    for item in TEST_ITEMS:
        organ, pathology = item[0], item[1]
        keywords = PATHOLOGY_KEYWORDS.get((organ, pathology))
        if keywords is None:
            continue

        col = f"{organ}_{pathology}"
        y_true, y_pred = [], []
        for _, row in df.iterrows():
            score = row[col]
            if score == "":
                continue
            patient_id = row["file_name"].replace(".nii.gz", "")
            organ_text = ground_truth.get(patient_id, {}).get(organ)
            if organ_text is None:
                continue
            normalized = organ_text.replace("-", " ")
            y_true.append(int(any(kw in normalized for kw in keywords)))
            y_pred.append(float(score))

        if not y_true:
            print(f"  {col}: no samples with both a prediction and ground-truth text")
            continue

        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)
        y_pred_bin = (y_pred_arr > 0.5).astype(int)
        accuracy = (y_true_arr == y_pred_bin).mean()
        base_rate = y_true_arr.mean()
        auc = roc_auc_score(y_true_arr, y_pred_arr) if len(set(y_true)) > 1 else float("nan")
        print(f"  {col}: n={len(y_true)} base_rate={base_rate:.3f} accuracy={accuracy:.3f} auc={auc:.3f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", default=_default_checkpoint(),
        help="path to a finetuned checkpoint_XXX.pth; defaults to the highest-numbered one in DEFAULT_OUTPUT_DIR",
    )
    parser.add_argument(
        "--untrained", action="store_true",
        help="evaluate the pre-finetuning baseline instead: expand_organs() applied to the "
             "released checkpoint with no finetuned weights loaded on top (FROZEN_ORGANS keep "
             "their real pretrained weights, the rest are freshly-initialized/untrained). "
             "Ignores --checkpoint.",
    )
    parser.add_argument(
        "--split", choices=["train", "valid"], default="valid",
        help="valid = the real held-out processed_valid_images/masks set; train was "
             "never actually held out from finetune.py's training data",
    )
    parser.add_argument("--output-dir", default=os.path.join(DEFAULT_OUTPUT_DIR, "eval_results"))
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None, help="for smoke-testing on a subset")
    args = parser.parse_args()
    if not args.untrained and args.checkpoint is None:
        parser.error("no checkpoint found under DEFAULT_OUTPUT_DIR and --checkpoint not given (or pass --untrained)")
    return args


@torch.inference_mode()
def evaluate():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_root = os.path.join(DATA_ROOT, f"processed_{args.split}_images")
    mask_root = os.path.join(DATA_ROOT, f"processed_{args.split}_masks")
    dataset = EvalOrganDataset(ORGANS, TEST_ITEMS, image_root=image_root, mask_root=mask_root)
    if args.max_samples is not None:
        dataset.img_paths = dataset.img_paths[:args.max_samples]
    if args.untrained:
        print(f"Evaluating on {len(dataset)} samples against the UNTRAINED baseline "
              f"(no finetuning at all - frozen organs keep pretrained weights, rest are random init)")
    else:
        print(f"Evaluating on {len(dataset)} samples against checkpoint {args.checkpoint}")
    print(f"{len(TEST_ITEMS)} organ-pathology test items, organs: {sorted(set(item[0] for item in TEST_ITEMS))}")

    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
        drop_last=False, collate_fn=collate_fn,
    )

    pad_func = transforms.DivisiblePadd(
        keys=["image", "label"], k=PATCH_SIZE, mode="constant", constant_values=0, method="end",
    )

    model = build_eval_model(None if args.untrained else args.checkpoint, device)
    text_feat_dict = model.prepare_text_feat(TEST_ITEMS)

    organ_feat_dict = {}
    results = []
    for image, mask, test_items, meta_info in tqdm(loader, desc="Infer"):
        fid = meta_info["file_name"]
        organ_feat_dict[fid] = {}

        image = image[None].to(device)
        mask = mask[None].to(device)

        test_organs = meta_info["test_organ_names"]
        whole_organ_sizes = dict(zip(
            test_organs,
            [torch.eq(mask, ORGANS.index(organ) + 1).sum().item() for organ in test_organs],
        ))
        test_organs = [organ for organ in test_organs if whole_organ_sizes[organ] > 0]
        test_items = [item for item in test_items if item[0] in test_organs]

        organ_logits = dict(zip(test_items, [[] for _ in test_items]))
        for k, v in list(organ_logits.items()):
            if len(v):
                continue  # already filled in by an earlier item sharing this organ
            organ_name = k[0]
            organ_id = ORGANS.index(organ_name)

            window_patch, window_mask = center_crop(
                image, torch.eq(mask, organ_id + 1), crop_size=CROP_SIZE,
            )
            window_mask = window_mask.float()
            window_mask[window_mask == 1] = organ_id + 1

            pad_data = pad_func({"image": window_patch[0], "label": window_mask[0]})
            window_patch, window_mask = pad_data["image"], pad_data["label"]

            organ_logits = model.forward_test_win(
                window_patch[None], window_mask[None], organ_logits, test_organs,
                text_feat_dict, organ_feat_dict[fid], whole_organ_sizes, skip_organ=organ_id,
            )

        row = [fid] + [""] * len(TEST_ITEMS)
        organ_logits = {item: probs for item, probs in organ_logits.items() if len(probs) > 0}
        for item, probs in organ_logits.items():
            row[TEST_ITEMS.index(item) + 1] = np.concatenate(probs).mean(0)[1]
        results.append(row)

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_name = "untrained" if args.untrained else os.path.splitext(os.path.basename(args.checkpoint))[0]
    out_path = os.path.join(args.output_dir, f"{ckpt_name}.csv")
    df = pd.DataFrame(results, columns=["file_name"] + ["_".join(item[:2]) for item in TEST_ITEMS])
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved {out_path}")

    scored = (df.iloc[:, 1:] != "").sum()
    print("evaluable-sample counts per test item (0 = organ/pathology never scored - "
          "check the data before trusting that column's numbers):")
    for item, count in scored.items():
        print(f"  {item}: {count}/{len(df)}")

    vqa_csv = vqa_abnormality_csv(args.split)
    if os.path.exists(vqa_csv):
        ground_truth = load_ground_truth(vqa_csv)
        score_against_ground_truth(df, ground_truth)
    else:
        print(f"\n{vqa_csv} not found - skipping ground-truth scoring.")


if __name__ == "__main__":
    evaluate()
