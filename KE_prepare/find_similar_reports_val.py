"""知识库构建第 2/2 步（validation 版）：给 validation 集里每个病人、每个器官算一个
IMAGE embedding（要用微调好的 checkpoint），再拿这个图像向量去 TRAIN 集的文本向量库
（compute_text_embeddings.py 在 train 上跑出来的 organ_report_embeddings.npz）里，
找最相似的 top-k 个 train 病人的报告。

跟 find_similar_reports.py（train 内部自检索）的区别：那边 bank 和 query 是同一批
train 病人，检索时要排除对角线（自己）；这边 bank 固定是 TRAIN 的文本库，query 是
VALIDATION 的图像，两拨病人天然不重叠，不存在"查到自己"的问题，所以相似度矩阵是
矩形的（n_query x n_bank），不用排除对角线。

检索方向仍然是"图像找文本"：validation 病人自己的报告是有的（RADGENOME_CSV 指向
validation_region_report.csv，用来当 ground truth caption），但检索用的 query 向量
用的是这个病人的 CT 影像，不是他自己的报告文本——这样跟真实推理时（还没写出报告）
的场景一致，同时也避免检索库里出现"用自己的报告问自己"这种数据泄漏。

依赖：先跑一次 compute_text_embeddings.py（finetune.py 的 PREPROCESSED_IMAGE_ROOT /
RADGENOME_CSV 指向 train 时）生成 TRAIN_TEXT_EMBEDDINGS 指向的那份 npz；再跑这个脚本
时，finetune.py 的这几个常量要指向 validation（PREPROCESSED_IMAGE_ROOT =
processed_valid_images，RADGENOME_CSV = validation_region_report.csv），因为
compute_image_embeddings/CTOrganDataset 都是直接读 finetune.py 里的这几个模块级常量。

输出：organ_annotation.json，长这样，每个病人一条记录：

    {"validation": [
        {
            "id": "valid_10000_a_1",
            "image_path": "valid_10000_a_1.nii.gz",
            "lung": "这个 validation 病人 lung 的 ground-truth caption 文本（清洗过的）",
            "lung_indices": [{"id": "train_10001_a_1", "organ": "lung"}, ...],   # 长度 = TOP_K，全部来自 TRAIN
            "heart": "...",
            "heart_indices": [...],
            ... 每个器官都有这么一对 "<organ>"/"<organ>_indices"
        },
        ...
    ]}

`"<organ>_indices"` 里存的是 `{"id": train 病人id, "organ": 器官名}`，不是 npz 里的
行号——原因跟 find_similar_reports.py 一样：不依赖 organ_report_embeddings.npz 当前
这次生成时的具体行排列。如果列表里某一项是 `null`，说明这个 (病人, 器官) 组合没有
image embedding 可用（比如这个病人这个器官压根没有 mask），没法检索。
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finetune import CTOrganDataset, DATA_ROOT, ORGANS, RADGENOME_CSV, build_organ_captions
from eval_finetune import DEFAULT_OUTPUT_DIR, build_eval_model
from compute_text_embeddings import clean_caption
from find_similar_reports import compute_image_embeddings

# bank 永远是 TRAIN 的文本库，不管这个脚本跑的时候 finetune.py 里的常量指向哪个
# split——跟 query（validation）用的 PREPROCESSED_IMAGE_ROOT/RADGENOME_CSV 完全分开。
TRAIN_TEXT_EMBEDDINGS = os.path.join(DATA_ROOT, "EK_files_train", "organ_report_embeddings.npz")
# 用第 40 个 epoch 的微调 checkpoint（不是自动挑最新的那个）——图像这一侧的 embedding
# 依赖 vision_projs 训练得好不好，跟文本那边不一样，必须显式指定用哪个 epoch。
FINETUNED_CHECKPOINT = os.path.join(DEFAULT_OUTPUT_DIR, "checkpoint_040.pth")
OUTPUT_DIR = os.path.join(DATA_ROOT, "EK_files_val")  # organ_annotation.json 存这里
JSON_KEY = "validation"
TOP_K = 5
MAX_SAMPLES = None  # 改成一个数字可以只跑一小部分病人，用于冒烟测试


def compute_top_k_indices(image_feats, image_valid, bank_text_feats, top_k, device):
    """按器官分别算：每个 validation 病人自己的 IMAGE embedding，跟**所有 train 病人**
    的 TEXT embedding 算余弦相似度（两边都已经 L2 归一化，直接点积就是余弦相似度），
    取 top_k 个最相似的。跟 find_similar_reports.py 的同名函数不同：这里 query
    （validation）和 bank（train）是两拨不重叠的病人，相似度矩阵是矩形的
    (n_query, n_bank)，不用排除对角线。

    这个器官没有 mask 的 validation 病人（image_valid 是 False）没有 image embedding
    可以拿来当查询向量，结果整行填 -1。
    """
    n_query = image_feats.shape[0]
    n_bank = bank_text_feats.shape[0]
    all_indices = np.full((n_query, len(ORGANS), top_k), -1, dtype=np.int64)

    for organ_idx, organ in enumerate(ORGANS):
        img = torch.from_numpy(image_feats[:, organ_idx, :]).to(device)  # (n_query, 256)
        txt = torch.from_numpy(bank_text_feats[:, organ_idx, :]).to(device)  # (n_bank, 256)

        # (n_query,256) @ (256,n_bank) = (n_query,n_bank)：两边都已经 L2 归一化过，
        # 矩阵乘法算出来的点积直接就是余弦相似度。sim[i,j] = 第i个 validation 病人的
        # 图像 跟 第j个 train 病人的文本 有多像。
        sim = img @ txt.t()

        top_k_idx = sim.topk(min(top_k, n_bank), dim=1).indices.cpu().numpy()

        organ_valid = image_valid[:, organ_idx]
        top_k_idx[~organ_valid] = -1

        all_indices[:, organ_idx, :] = top_k_idx
    return all_indices


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    bank_npz = np.load(TRAIN_TEXT_EMBEDDINGS)
    bank_text_feats = bank_npz["feats"]
    bank_patient_ids = [str(p) for p in bank_npz["patient_ids"]]
    assert list(bank_npz["organs"]) == ORGANS, "organ order in bank npz doesn't match current ORGANS"
    print(f"bank: {len(bank_patient_ids)} train patients, {len(ORGANS)} organs")
    print(f"using finetuned checkpoint: {FINETUNED_CHECKPOINT}")

    # RADGENOME_CSV/PREPROCESSED_IMAGE_ROOT/PREPROCESSED_MASK_ROOT 这几个 finetune.py
    # 里的常量此时要指向 validation，query 病人列表才会是 validation 集——跟
    # compute_text_embeddings.py 用同一个 CTOrganDataset 过滤逻辑，保证跟训练/评估时
    # 用的 validation 病人列表一致。
    captions = build_organ_captions(RADGENOME_CSV, ORGANS)
    dataset = CTOrganDataset(ORGANS, captions, vis_processor=None)
    samples = dataset.samples
    if MAX_SAMPLES is not None:
        samples = samples[:MAX_SAMPLES]
    query_patient_ids = [sample_id for _, _, sample_id in samples]
    print(f"query: {len(query_patient_ids)} validation patients")

    model = build_eval_model(FINETUNED_CHECKPOINT, device)

    image_feats, image_valid = compute_image_embeddings(model, query_patient_ids, device)
    print(f"image embeddings computed for {image_valid.sum()}/{image_valid.size} (patient, organ) pairs")

    top_k_indices = compute_top_k_indices(image_feats, image_valid, bank_text_feats, TOP_K, device)

    records = []
    for i, pid in enumerate(query_patient_ids):
        record = {"id": pid, "image_path": f"{pid}.nii.gz"}
        for organ_idx, organ in enumerate(ORGANS):
            record[organ] = clean_caption(captions[pid][organ], organ)
            # 存 (id, organ) 而不是 npz 里的行号，原因见模块开头的说明。-1（没有可用
            # image embedding）转成 null，不能直接拿 -1 去 bank_patient_ids 里索引
            # （那样会悄悄取到最后一个病人，不是"没有匹配"的意思）。
            record[f"{organ}_indices"] = [
                {"id": bank_patient_ids[idx], "organ": organ} if idx >= 0 else None
                for idx in top_k_indices[i, organ_idx].tolist()
            ]
        records.append(record)
    assert len(records) == len(query_patient_ids), "record count doesn't match patient count"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    json_path = os.path.join(OUTPUT_DIR, "organ_annotation.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({JSON_KEY: records}, f, ensure_ascii=False)
    print(f"Saved {json_path}: {len(records)} records")


if __name__ == "__main__":
    main()