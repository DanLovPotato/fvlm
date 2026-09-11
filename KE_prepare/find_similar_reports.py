"""知识库构建第 2/2 步：给每个病人、每个器官算一个 IMAGE embedding（要用微调好的
checkpoint），再拿这个图像向量去 compute_text_embeddings.py 生成的文本向量库
（organ_report_embeddings.npz）里，找最相似的 top-k 个"其他病人"的报告。

检索方向是"图像找文本"：真实用的时候，病人自己的报告还没写出来
（这正是我们要生成报告的目的），没法拿病人自己的报告文本去检索；能用的只有他的 CT
影像。所以每个器官的检索 query，是**这个病人自己**的器官 IMAGE embedding（用模型
推理时的裁剪流程算出来的——就是 eval_finetune.py 那套 center_crop +
forward_test_win），拿去跟**其他病人**的器官 TEXT embedding 比对。

输出：organ_annotation.json，长这样，每个病人一条记录：

    {"train": [
        {
            "id": "train_10000_a_1",
            "image_path": "train_10000_a_1.nii.gz",
            "lung": "这个病人 lung 的 caption 文本（清洗过的）",
            "lung_indices": [{"id": "train_10001_a_1", "organ": "lung"}, ...],   # 长度 = TOP_K
            "heart": "...",
            "heart_indices": [...],
            ... 每个器官都有这么一对 "<organ>"/"<organ>_indices"
        },
        ...
    ]}

`"<organ>_indices"` 里存的是 `{"id": 病人id, "organ": 器官名}`，不是 npz 里的行号——
故意不存行号，是因为行号绑定在生成这份 json 那一刻 `organ_report_embeddings.npz`
具体的行排列上，那份 npz 以后只要重新生成一次（哪怕内容不变，顺序变了），行号就全部
悄悄指错地方，还不会报错。存 `(id, organ)` 这两个字符串就不依赖 npz 的具体排列，
想找到实际 embedding 的话，去 `organ_report_embeddings.npz` 里按 `id` 找到对应的行、
再按 `organ` 找到对应的列就行。这些是跟当前病人自己的 IMAGE embedding 余弦相似度
从高到低排的 top-k 个（已经排除病人自己，不会检索到自己）。如果列表里某一项是
`null`，说明这个 (病人, 器官) 组合没有 image embedding 可用（比如这个病人这个器官
压根没有 mask），没法检索。
"""
import json
import os
import sys

import numpy as np
import torch
from monai import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finetune import CROP_SIZE, DATA_ROOT, ORGAN_MASK_ID, ORGANS, PATCH_SIZE, PREPROCESSED_IMAGE_ROOT, PREPROCESSED_MASK_ROOT, RADGENOME_CSV, build_organ_captions
from eval_finetune import DEFAULT_OUTPUT_DIR, build_eval_model, center_crop
from compute_text_embeddings import clean_caption

TEXT_EMBEDDINGS = os.path.join(DATA_ROOT, "EK_files_train", "organ_report_embeddings.npz")  # compute_text_embeddings.py 的输出
# 用第 40 个 epoch 的微调 checkpoint（不是自动挑最新的那个）——图像这一侧的 embedding
# 依赖 vision_projs 训练得好不好，跟文本那边不一样，必须显式指定用哪个 epoch。
FINETUNED_CHECKPOINT = os.path.join(DEFAULT_OUTPUT_DIR, "checkpoint_040.pth")
OUTPUT_DIR = os.path.join(DATA_ROOT, "EK_files_train")  # organ_annotation.json 存这里
TOP_K = 5
MAX_SAMPLES = None  # 改成一个数字可以只跑一小部分病人，用于冒烟测试


@torch.inference_mode()
def compute_image_embeddings(model, patient_ids, device):
    """给每个病人、每个器官算一个 256 维的图像 embedding：围绕这个器官的 mask 裁剪、
    pad，然后走跟 `forward_test_win` 在别处一样的那条路径（ROI 池化 -> attention ->
    vision_proj，见 blip_pretrain.py:501-524）。

    是按"每个器官"跑一次 ViT 前向，不是按"每个病人"跑一次——每个器官都要单独裁一个
    窗口，所以一个病人如果 10 个器官都在，最多要跑 10 次前向，比只跑一次贵不少；
    `eval_finetune.py` 自己评估时对每个验证样本也是这么做的，不是这里特殊。

    调用 `forward_test_win` 时 `organ_logits`/`text_feat_dict` 都传空字典：我们只要
    它顺手写进 `organ_feat_dict` 里的图像 embedding，不需要它去跟文本比较算 logits，
    传空字典正好让它内部那段"跟文本比较"的逻辑什么也不做，直接跳过。
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

        # 每个器官单独裁一个窗口，center crop， 算包围盒 → 围绕包围盒裁窗口 → pad 到 patch 整数倍
        # 用 ORGAN_MASK_ID 而不是 ORGANS.index(organ)+1：'pleura' 在磁盘 mask 上没有
        # 自己的 seg value，读的是 'lung' 的（见 finetune.py 模块开头的说明），
        # ORGANS.index("pleura")+1 会指向一个磁盘上根本不存在的 seg value。
        whole_organ_sizes = {
            organ: torch.eq(mask, ORGAN_MASK_ID[organ]).sum().item() for organ in ORGANS
        }
        organ_feat_dict = {}
        for organ_id, organ in enumerate(ORGANS):
            if whole_organ_sizes[organ] == 0:
                continue  # this patient's preprocessed mask doesn't have this organ at all

            mask_value = ORGAN_MASK_ID[organ]
            window_patch, window_mask = center_crop(image, torch.eq(mask, mask_value), crop_size=CROP_SIZE)
            window_mask = window_mask.float()
            window_mask[window_mask == 1] = mask_value
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
    """按器官分别算：每个病人自己的 IMAGE embedding，跟**所有其他病人**的 TEXT
    embedding 算余弦相似度（两边都已经 L2 归一化，直接点积就是余弦相似度），取
    top_k 个最相似的，排除病人自己那一行。每个器官单独算一个 N×N 的相似度矩阵
    （N=24116 时大约 2.3GB float32）

    这个器官没有 mask 的病人（image_valid 是 False）没有 image embedding 可以拿来
    当查询向量，结果整行填 -1。
    """
    n = text_feats.shape[0]  # 病人数，24116
    # 先建一个全 -1 的占位数组，形状 (病人数, 10个器官, top_k) —— 后面算出真实结果的
    # 地方才会覆盖成真的行号，算不出来的（比如这个器官没 mask）就保持 -1。
    all_indices = np.full((n, len(ORGANS), top_k), -1, dtype=np.int64)

    for organ_idx, organ in enumerate(tqdm(ORGANS, desc="top-k neighbors")):
        # 只拿"这一个器官"的列：所有病人的 IMAGE embedding、所有病人的 TEXT embedding，
        # 形状都是 (n, 256)。同一个 organ_idx，保证不会跨器官比较。
        img = torch.from_numpy(image_feats[:, organ_idx, :]).to(device)
        txt = torch.from_numpy(text_feats[:, organ_idx, :]).to(device)

        # (n,256) @ (256,n) = (n,n)：因为两边都已经 L2 归一化过，矩阵乘法算出来的
        # 点积直接就是余弦相似度。sim[i,j] = 第i个病人的图像 跟 第j个病人的文本 有多像。
        sim = img @ txt.t()

        # 对角线 sim[i,i] 是"这个病人自己的图像 vs 自己的文本"，检索时要排除掉
        # （不能检索到自己），设成 -inf 这样 topk 永远不会选到它。
        sim.fill_diagonal_(float("-inf"))

        # 每一行（每个病人）取相似度最高的 top_k 个列号（也就是最相似的 top_k 个
        # "其他病人"在 n 个病人里的位置）。.indices 拿的是位置下标，不是相似度分数本身。
        top_k_idx = sim.topk(top_k, dim=1).indices.cpu().numpy()

        # 但如果这个病人自己在这个器官上根本没有有效的 image embedding
        # （image_valid 是 False，比如这个病人这个器官没 mask），那前面算出来的
        # top_k_idx 是拿一个没意义的查询向量（全零）检索出来的，不能要——
        # 把这些行整行覆盖成 -1，表示"没法检索"。
        organ_valid = image_valid[:, organ_idx]
        top_k_idx[~organ_valid] = -1

        # 把这个器官算出来的结果，填进最终结果数组对应的这一列。
        all_indices[:, organ_idx, :] = top_k_idx
    return all_indices


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    npz = np.load(TEXT_EMBEDDINGS)
    text_feats = npz["feats"]
    patient_ids = [str(p) for p in npz["patient_ids"]]
    assert list(npz["organs"]) == ORGANS, "organ order in npz doesn't match current ORGANS"
    if MAX_SAMPLES is not None:
        patient_ids = patient_ids[:MAX_SAMPLES]
        text_feats = text_feats[:MAX_SAMPLES]
    print(f"{len(patient_ids)} patients, {len(ORGANS)} organs")
    print(f"using finetuned checkpoint: {FINETUNED_CHECKPOINT}")

    captions = build_organ_captions(RADGENOME_CSV, ORGANS)

    model = build_eval_model(FINETUNED_CHECKPOINT, device)

    image_feats, image_valid = compute_image_embeddings(model, patient_ids, device)
    print(f"image embeddings computed for {image_valid.sum()}/{image_valid.size} (patient, organ) pairs")

    top_k_indices = compute_top_k_indices(image_feats, image_valid, text_feats, TOP_K, device)

    records = []
    for i, pid in enumerate(patient_ids):
        record = {"id": pid, "image_path": f"{pid}.nii.gz"}
        for organ_idx, organ in enumerate(ORGANS):
            record[organ] = clean_caption(captions[pid][organ], organ)
            # 存 (id, organ) 而不是 npz 里的行号，不依赖 organ_report_embeddings.npz
            # 当前这次生成时的具体行排列 - 见模块开头的说明。-1（没有可用 image
            # embedding）转成 null，不能直接拿 -1 去 patient_ids 里索引（那样会
            # 悄悄取到最后一个病人，不是"没有匹配"的意思）。
            record[f"{organ}_indices"] = [
                {"id": patient_ids[idx], "organ": organ} if idx >= 0 else None
                for idx in top_k_indices[i, organ_idx].tolist()
            ]
        records.append(record)
    assert len(records) == len(patient_ids), "record count doesn't match patient count"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    json_path = os.path.join(OUTPUT_DIR, "organ_annotation.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"train": records}, f, ensure_ascii=False)
    print(f"Saved {json_path}: {len(records)} records")


if __name__ == "__main__":
    main()
