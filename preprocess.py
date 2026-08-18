"""Offline preprocessing: crop each raw CT volume to the bounding box of its organ
masks (+ margin), resample to the reference spacing the pretrained model's window is
calibrated against, then pad to the fixed (112, 256, 352) input size if it's smaller
than that, and save to disk.
Run this once before finetune.py - finetune.py's dataset only reads this step's
output, never the raw volumes directly.

The crop/pad/save core below is copied from fvlm_original/data/preprocess.py's
process_image() unchanged (same extend_d/extend_hw margin, same SpatialPadd call, same
sanity assert that organ ids survive the crop, same SaveImaged pattern). The front end
differs because it has to: these raw files aren't laid out under the original's
train_mask/train_fix naming convention (iter_samples() below matches the actual
IMAGE_ROOT layout instead), and the masks come as one separate {organ}.nii.gz file per
organ rather than a single pre-merged label volume (merge_organ_masks() combines them
into that same one-integer-segmentation encoding the original assumes as input, and
what BlipPretrain.forward() indexes into via self.organs[cl_organ_id]).

fvlm_original/data/resize.py resamples every volume to REF_SPACING=(1.0, 1.0, 3.0) mm
before data/preprocess.py's crop/pad ever runs, so CROP_SIZE=(112, 256, 352) is
calibrated against 3mm slices, not 1mm. These raw volumes are isotropic 1mm on every
axis instead, so process_sample resamples to REF_SPACING first (trilinear for image,
nearest for label, matching resize.py's Step 2), before cropping/padding to CROP_SIZE -
otherwise the depth axis's 112-voxel budget would only cover 112mm of real anatomy
instead of the 336mm the pretrained model's window assumes.

data/ 文件夹里一共 4 个脚本,原版的完整流水线是:fix_data.py → generate_mask.py → resize.py → preprocess.py

- 合并 mask:merge_organ_masks()
    把这个病人 10 个独立的 {organ}.nii.gz 二值文件,
    合成一张单通道多类别体积(体素值 0=背景,1..10=对应 ORGANS 里的器官)。
- 重采样到参考 spacing:
    在裁剪之前,先把图像和合并后的 mask 一起重采样到 REF_SPACING=(1,1,3),
    对齐预训练模型窗口校准时假设的体素间距(尤其是深度轴,原始 1mm 会被
    压缩到约 1/3 的体素数)。
- 裁剪掉不含器官的区域:
    用这张合并 mask 找出"所有 10 个器官加起来占据哪个立方体范围"(np.nonzero
    求 bbox),往外扩一点边距(深度±5,长宽±20),然后把 CT 图像和 mask 都
    裁到这个范围——范围外的(比如四肢、头部这类跟这 10 个器官完全无关的部分)
    直接丢掉,不进入后续训练。
- 再 pad 成固定尺寸 (112, 256, 352)——
    因为每个病人器官分布范围大小不一样,裁出来的尺寸不统一,模型的 ViT 需要
    固定输入尺寸,所以裁完再补零到统一大小。注意:pad 只会往大补、不会往小
    裁——如果裁剪后的范围本身就比 (112,256,352) 大(比如 bone/lung/
    abdomen 这几个偏大的器官经常会这样),存盘文件会比这个尺寸更大,真正裁到
    固定尺寸是训练时 finetune.py 里的随机裁剪(RandSpatialCropd)做的。
- 存到 processed_images/processed_masks,
    供 finetune.py 的 CTOrganDataset 直接读取(不再碰原始的 preprocessed/region_mask)。
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import torch
from monai import transforms
from tqdm import tqdm

DATA_ROOT = r"/mnt/researchdrive/ptiwari9/Staff_Trainee_Folders/Dan/chestCT/data/dataset"
# Train defaults - overridden in __main__ below for --split valid. Kept as plain
# module-level names (not wrapped in a function) so iter_samples()/process_sample()
# below can keep referring to them as-is, matching this module's existing style.
IMAGE_ROOT = os.path.join(DATA_ROOT, "train_preprocessed")
MASK_ROOT = os.path.join(DATA_ROOT, "train_region_mask")
PREPROCESSED_IMAGE_ROOT = os.path.join(DATA_ROOT, "processed_train_images")
PREPROCESSED_MASK_ROOT = os.path.join(DATA_ROOT, "processed_train_masks")

ORGANS = [
    "abdomen", "bone", "breast", "esophagus", "heart",
    "lung", "mediastinum", "thyroid", "trachea and bronchie"
]

CROP_SIZE = (112, 256, 352)
# Same margin fvlm_original/data/preprocess.py uses around the union organ-mask
# bounding box before padding/cropping to CROP_SIZE.
EXTEND_D = 5
EXTEND_HW = 20

# Matches fvlm_original/data/resize.py's ref_spacing exactly - CROP_SIZE's depth axis
# (112 voxels) only makes sense as the ~336mm window the pretrained model expects if
# every volume is resampled to this spacing first.
REF_SPACING = (1.0, 1.0, 3.0)


def iter_samples():
    for patient_dir in sorted(os.listdir(IMAGE_ROOT)):
        patient_path = os.path.join(IMAGE_ROOT, patient_dir)
        if not os.path.isdir(patient_path):
            continue
        for study_dir in sorted(os.listdir(patient_path)):
            study_path = os.path.join(patient_path, study_dir)
            if not os.path.isdir(study_path):
                continue
            for file_name in sorted(os.listdir(study_path)):
                if file_name.endswith(".nii.gz"):
                    sample_id = file_name[:-len(".nii.gz")]
                    yield os.path.join(study_path, file_name), sample_id


def merge_organ_masks(mask_dir, mask_loader):
    """Load each organ's separate {organ}.nii.gz mask and merge into a single label
    volume, id = ORGANS.index(organ) + 1 - the one-integer-segmentation encoding
    fvlm_original's own preprocess.py assumes as input.

    Starts from a clone of the first organ mask found (rather than a bare
    torch.zeros(...)) specifically so the result stays a MONAI MetaTensor with real
    affine/spacing metadata carried over from an actual loaded file - SaveImaged
    below needs that metadata to write a valid, correctly-oriented .nii.gz.
    """
    merged = None
    for organ_id, organ in enumerate(ORGANS):
        mask_path = os.path.join(mask_dir, f"{organ}.nii.gz")
        if not os.path.exists(mask_path):
            continue
        organ_mask = mask_loader({"label": mask_path})["label"]
        if merged is None:
            merged = organ_mask.clone()
            merged[:] = 0
        merged[organ_mask > 0] = organ_id + 1
    return merged


def process_sample(loader, mask_loader, finalize, sample):
    img_path, sample_id = sample
    try:
        out_img_path = os.path.join(PREPROCESSED_IMAGE_ROOT, f"{sample_id}.nii.gz")
        out_label_path = os.path.join(PREPROCESSED_MASK_ROOT, f"{sample_id}.nii.gz")
        if Path(out_img_path).exists() and Path(out_label_path).exists():
            return None

        mask_dir = os.path.join(MASK_ROOT, f"seg_{sample_id}")
        if not os.path.isdir(mask_dir):
            return None

        label = merge_organ_masks(mask_dir, mask_loader)
        if label is None:
            # none of this sample's organ mask files exist - nothing to crop to
            return None

        data = loader({"image": img_path})
        image = data["image"]
        # filename_or_obj drives SaveImaged's output basename below; point the
        # (merged, not loaded-from-a-single-file) label at the same sample_id the
        # image uses so both outputs land under the same name in their own dirs.
        label.meta["filename_or_obj"] = img_path
        data["label"] = label

        # Resample to REF_SPACING before anything else, in the volume's native (not
        # yet transposed) axis order - matches resize.py, which computes target size
        # from the image's own affine (mask/image share the same grid, so the same
        # target_size applies to both) and resamples image/label before they're ever
        # combined with the crop/pad/transpose logic below.
        affine = image.meta["affine"]
        spacing = tuple(abs(affine[i, i].item()) for i in range(3))
        _, x, y, z = image.shape
        scale = [spacing[i] / REF_SPACING[i] for i in range(3)]
        target_size = [int(x * scale[0]), int(y * scale[1]), int(z * scale[2])]

        # 这两行就是真正执行重采样到 (1,1,3)
        # image 用 trilinear(三线性插值)——图像是连续的灰度值,插值产生的中间值是合理的。
        # label 用 nearest(最近邻)——mask 里的体素值是离散的器官编号(0,1,2...10),
        # 如果用三线性插值会插出 3.7 这种不存在的"器官编号",所以必须用最近邻,保证结果还是原来那几个整数值之一。
        resample = transforms.Compose(
            [
                transforms.Resized(keys=["image"], spatial_size=target_size, mode="trilinear"),
                transforms.Resized(keys=["label"], spatial_size=target_size, mode="nearest"),
            ]
        )
        data = resample(data)

        data = finalize(data)
        image = data["image"]
        label = data["label"]

        old_unique_organ_ids = label.unique()

        roi_coords = np.nonzero(label[0])
        min_dhw = torch.from_numpy(np.min(roi_coords, axis=1))
        max_dhw = torch.from_numpy(np.max(roi_coords, axis=1))

        min_dhw = torch.maximum(
            min_dhw - torch.tensor([EXTEND_D, EXTEND_HW, EXTEND_HW]),
            torch.tensor([0, 0, 0]),
        )
        max_dhw = torch.minimum(
            max_dhw + torch.tensor([EXTEND_D, EXTEND_HW, EXTEND_HW]),
            torch.tensor([image.shape[1], image.shape[2], image.shape[3]]),
        )

        data["image"] = image[
            :, min_dhw[0]:max_dhw[0], min_dhw[1]:max_dhw[1], min_dhw[2]:max_dhw[2]
        ]
        data["label"] = label[
            :, min_dhw[0]:max_dhw[0], min_dhw[1]:max_dhw[1], min_dhw[2]:max_dhw[2]
        ]

        new_unique_organ_ids = data["label"].unique()
        assert torch.all(old_unique_organ_ids == new_unique_organ_ids)

        saver = transforms.Compose(
            [
                transforms.SpatialPadd(
                    keys=["image"], spatial_size=CROP_SIZE,
                    mode="constant", constant_values=0,
                ),
                transforms.SpatialPadd(
                    keys=["label"], spatial_size=CROP_SIZE,
                    mode="constant", constant_values=0,
                ),
                transforms.SaveImaged(
                    output_dir=PREPROCESSED_IMAGE_ROOT, keys=["image"],
                    output_postfix="", separate_folder=False, resample=False,
                ),
                transforms.SaveImaged(
                    output_dir=PREPROCESSED_MASK_ROOT, keys=["label"],
                    output_postfix="", separate_folder=False, resample=False,
                ),
            ]
        )
        saver(data)

    except Exception as e:
        print(e, img_path)
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "valid"], default="train")
    args = parser.parse_args()

    IMAGE_ROOT = os.path.join(DATA_ROOT, f"{args.split}_preprocessed")
    MASK_ROOT = os.path.join(DATA_ROOT, f"{args.split}_region_mask")
    PREPROCESSED_IMAGE_ROOT = os.path.join(DATA_ROOT, f"processed_{args.split}_images")
    PREPROCESSED_MASK_ROOT = os.path.join(DATA_ROOT, f"processed_{args.split}_masks")

    os.makedirs(PREPROCESSED_IMAGE_ROOT, exist_ok=True)
    os.makedirs(PREPROCESSED_MASK_ROOT, exist_ok=True)

    # Load-only, in each file's native (not yet transposed) axis order - the resample
    # step in process_sample needs the image's own affine before anything reorders or
    # rescales it.
    loader = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"], image_only=True, ensure_channel_first=True),
        ]
    )
    mask_loader = transforms.Compose(
        [
            transforms.LoadImaged(keys=["label"], image_only=True, ensure_channel_first=True),
        ]
    )
    # Runs after the REF_SPACING resample in process_sample. Matches
    # fvlm_original/data/preprocess.py's loader exactly (transpose then intensity
    # window), so the encoder sees the same axis order and intensity distribution it
    # was trained on.
    # !!! It does two things: 1. transpose 2. intensity normalization(intensity windowing) !!!
    # Transposed——把 (C,X,Y,Z) 的轴顺序重排成 (C,Z,Y,X),也就是后面代码统一按 (D,H,W) 使用的顺序。
    # 只是重新排列维度顺序,不判断方向、不改动任何体素数值。对 image 和 label 都做(两者要保持同步对齐)。
    # ScaleIntensityRanged——只对 image 做:把 HU 值裁到 [-1150, 350] 区间,再线性映射到 [0.0, 1.0]。
    # label 是整数器官编号,不能做强度变换,所以只写了 keys=["image"]。
    finalize = transforms.Compose(
        [
            transforms.Transposed(keys=["image", "label"], indices=(0, 3, 2, 1)),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-1150, a_max=350,
                b_min=0.0, b_max=1.0, clip=True,
            ),
        ]
    )

    samples = list(iter_samples())

    max_workers = 16
    func = partial(process_sample, loader, mask_loader, finalize)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for _ in tqdm(executor.map(func, samples), total=len(samples)):
            pass
