"""KE_prepare_unique 第 3 步：给每个 train 病人、每个器官算一个 IMAGE embedding
（要用微调好的 checkpoint），再拿这个图像向量去 2_compute_text_embeddings.py 生成的
"去重后的文本向量库"（organ_report_embeddings.npz）里，找最相似的 top-k 个"其他
病人的报告"。

跟 KE_prepare/find_similar_reports.py（旧版，bank 按病人建）的区别：
- 旧版 bank 是"每个病人一行"，query 和 bank 是同一批病人、同一个下标体系，排除自己
  就是排除相似度矩阵的对角线（sim[i, i]）。
- 这里 bank 是"每个器官下去重后的每条唯一文本一行"（2_compute_text_embeddings.py
  的输出），跟病人数完全不是一回事：相似度矩阵是矩形的
  (n_train_patients, n_unique_texts_of_organ)，跟 KE_prepare/find_similar_reports_val.py
  用 TRAIN 文本库检索 VALIDATION 图像时的矩形矩阵写法一样（直接照抄了它的
  compute_top_k_indices 结构），区别是那边 query/bank 天然不重叠不用排除自己，这里
  query（train）跟 bank（也是从 train 的 caption 去重来的）有重叠，必须排除自己。

排除自己的方式不是排除对角线（矩阵不是方的，没有对角线可言），
而是：先查这个病人
自己在这个器官下清洗后的 caption 文本，在 bank 里对应哪一行（build_self_row_idx()
按文本内容查——去重就是按文本分组的，同一段文本在同一个器官下必然只出现在 bank 的
一行里，不会出现"我自己的文本分散在两行"这种情况），再把那一行的相似度设成 -inf，
让 topk 不会选到它。这样即使这段文本被成千上万个病人共享（比如某个器官的模板句
"lung shows no significant abnormalities."），也只是排除"跟当前查询病人完全同属一行"
这一整行，其余病人查询时这一行仍然可以被选中——不是"这条模板从 bank 里彻底消失"，
只是对每个查询病人各自屏蔽自己所在的那一行。屏蔽之后，bank 里剩下的每一行都保证
不包含当前查询病人（因为一个病人在同一个器官下只能属于唯一一行），所以从命中的行里
随便挑一个 patient_id 当代表都不会挑到查询病人自己。

输出：organ_annotation.json，跟旧版格式保持一致，每个病人一条记录：

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

`"<organ>_indices"` 里存的是 `{"id": 病人id, "organ": 器官名}`：bank 现在按"去重
文本"存，检索命中的是 bank 里的第 j 行文本、不是某一个特定病人，这里从那一行的
patient_ids 列表（来自 unique_reports.json，见上一段说明为什么这里保证不会是查询
病人自己）里挑第一个病人 id 当代表，跟旧版一样只存 (id, organ) 两个字符串、不存 npz
行号，不依赖 organ_report_embeddings.npz 当前这次生成的具体行排列。如果某一项是
`null`，说明这个 (病人, 器官) 组合没有 image embedding 可用（比如这个病人这个器官
压根没有 mask），没法检索；如果某个器官排除自己之后 bank 剩下的行数比 TOP_K 还少
（目前 10 个器官都远超 TOP_K=5，理论上才会发生），多出来的槎位也填 null。

注意：patient_ids[0] 固定取的是"最早出现这条文本的病人"，不是随机选的，但也不是
"最相似的病人"——相似度排序只排到"文本"这一层，具体代表哪个病人跟查询病人像不像
无关。
"""
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

_FVLM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _FVLM_ROOT)
sys.path.insert(0, os.path.join(_FVLM_ROOT, "KE_prepare"))

from finetune import CTOrganDataset, DATA_ROOT, ORGANS, RADGENOME_CSV, build_organ_captions
from eval_finetune import DEFAULT_OUTPUT_DIR, build_eval_model
from compute_text_embeddings import clean_caption
# 图像 embedding 那部分（center crop + forward_test_win）跟"bank 是按病人存还是按
# 去重文本存"完全无关，直接复用旧版实现，不重新抄一遍。
from find_similar_reports import compute_image_embeddings

# 复用 1_get_unique_report.py / 2_compute_text_embeddings.py 的输出目录：既是这一步的
# 输入（unique_reports.json + organ_report_embeddings.npz），也是这一步的输出
# （organ_annotation.json），三个文件放在一起，配套使用不容易弄混版本。
UNIQUE_DIR = os.path.join(DATA_ROOT, "EK_files_train_unique")
UNIQUE_REPORTS_JSON = os.path.join(UNIQUE_DIR, "unique_reports.json")
TEXT_EMBEDDINGS = os.path.join(UNIQUE_DIR, "organ_report_embeddings.npz")
OUTPUT_DIR = UNIQUE_DIR
FINETUNED_CHECKPOINT = os.path.join(DEFAULT_OUTPUT_DIR, "checkpoint_040.pth")
TOP_K = 5
MAX_SAMPLES = None  # 改成一个数字可以只跑一小部分病人，用于冒烟测试


def build_self_row_idx(unique_reports, patient_ids, captions):
    """对每个器官，算出每个病人自己那条（清洗后的）caption 在 bank 里对应第几行。

    用文本内容去查，不是用 patient_ids 列表去查——因为去重就是按"同一个器官下文本
    是否完全相同"分组的，所以一个病人在某个器官下清洗后的 caption，在 bank 里必然
    对应且只对应一行（unique_reports.json 里那个器官的列表本身就不会有重复文本）。
    如果查不到，说明这次用的 unique_reports.json/organ_report_embeddings.npz 跟当前
    这批病人的 caption 不是同一批数据生成的，直接 assert 报错，不能悄悄跳过。
    """
    self_row_idx = {}
    for organ in tqdm(ORGANS, desc="locate self rows"):
        text_to_row = {entry["text"]: i for i, entry in enumerate(unique_reports[organ])}
        assert len(text_to_row) == len(unique_reports[organ]), (
            f"{organ}: unique_reports.json 里出现了重复文本，去重没有做干净"
        )
        rows = np.empty(len(patient_ids), dtype=np.int64)
        for i, pid in enumerate(patient_ids):
            text = clean_caption(captions[pid][organ], organ)
            assert text in text_to_row, (
                f"{pid} 在 {organ} 下清洗后的 caption 在 unique_reports.json 里找不到匹配的行——"
                "这份 npz/json 可能不是用当前这批病人的 caption 生成的，需要重新跑 1/2 两步"
            )
            rows[i] = text_to_row[text]
        self_row_idx[organ] = rows
    return self_row_idx


def compute_top_k_indices(image_feats, image_valid, bank_text_feats, self_row_idx, top_k, device):
    """按器官分别算：每个病人自己的 IMAGE embedding，跟**这个器官去重后的所有 bank
    文本**算余弦相似度（两边都已经 L2 归一化，直接点积就是余弦相似度），排除掉
    "跟自己同属一行"的那一行之后，取 top_k 个最相似的。

    跟 KE_prepare/find_similar_reports.py 的同名函数不同：那边矩阵是方的
    (n_patients, n_patients)，排除自己是 fill_diagonal_；这里矩阵是矩形的
    (n_patients, n_bank_of_this_organ)，排除自己要按每个病人各自的 self_row_idx
    单独定位、单独设成 -inf，不能用对角线那一套。

    这个器官没有 mask 的病人（image_valid 是 False）没有 image embedding 可以拿来
    当查询向量，结果整行填 -1。
    """
    n_patients = image_feats.shape[0]
    all_indices = np.full((n_patients, len(ORGANS), top_k), -1, dtype=np.int64)

    for organ_idx, organ in enumerate(tqdm(ORGANS, desc="top-k neighbors")):
        img = torch.from_numpy(image_feats[:, organ_idx, :]).to(device)  # (n_patients, 256)
        txt = torch.from_numpy(bank_text_feats[organ]).to(device)       # (n_bank, 256)
        n_bank = txt.shape[0]

        # (n_patients,256) @ (256,n_bank) = (n_patients,n_bank)：两边都已经 L2 归一化
        # 过，矩阵乘法算出来的点积直接就是余弦相似度。
        # sim[i,j] = 第i个病人的图像 跟 bank 第j行去重文本 有多像。
        sim = img @ txt.t()

        # 把每个病人自己那一行（在这个器官 bank 里的行号）的相似度设成 -inf——不是
        # 对角线，是按每个病人各自查到的行号单独定位（见 build_self_row_idx）。
        rows = torch.arange(n_patients, device=device)
        cols = torch.from_numpy(self_row_idx[organ]).to(device)
        sim[rows, cols] = float("-inf")

        # 排除自己那一行之后，最多还剩 n_bank - 1 个候选可选，防止 top_k 比这还大时
        # torch.topk 报错（目前 10 个器官都远超 TOP_K=5，实际不会触发，但留个保护）。
        k = min(top_k, n_bank - 1)
        top_k_idx = sim.topk(k, dim=1).indices.cpu().numpy()  # (n_patients, k)

        # 这个病人自己在这个器官上根本没有有效 image embedding（image_valid 是
        # False，比如没有 mask）：前面算出来的 top_k_idx 是拿一个没意义的查询向量
        # （全零）检索出来的，不能要——整行覆盖成 -1，表示"没法检索"。
        organ_valid = image_valid[:, organ_idx]
        top_k_idx[~organ_valid] = -1

        all_indices[:, organ_idx, :k] = top_k_idx
        # k < top_k 时（目前不会发生），剩下的槎位保持初始化时的 -1。
    return all_indices


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(UNIQUE_REPORTS_JSON, "r", encoding="utf-8") as f:
        unique_reports = json.load(f)

    npz = np.load(TEXT_EMBEDDINGS)
    assert list(npz["organs"]) == ORGANS, "organ order in npz doesn't match current ORGANS"
    bank_text_feats = {organ: npz[f"{organ}_feats"] for organ in ORGANS}
    for organ in ORGANS:
        # npz 里每个器官的行数应该跟 unique_reports.json 里这个器官的条目数完全一样——
        # 两份文件本该是同一次 pipeline 跑出来的，行数不一致说明其中一个是过期版本。
        assert bank_text_feats[organ].shape[0] == len(unique_reports[organ]), (
            f"{organ}: npz 里 {bank_text_feats[organ].shape[0]} 行，"
            f"unique_reports.json 里却有 {len(unique_reports[organ])} 条——两者对不上，"
            "是不是其中一个文件是过期版本？需要重新跑 1_get_unique_report.py / "
            "2_compute_text_embeddings.py"
        )

    # 跟 1/2 两步用同一套 CTOrganDataset 过滤逻辑（image、mask、caption 三者都存在），
    # 保证这里检索的病人集合跟建 bank 时的病人集合完全一致。
    captions = build_organ_captions(RADGENOME_CSV, ORGANS)
    dataset = CTOrganDataset(ORGANS, captions, vis_processor=None)
    samples = dataset.samples
    if MAX_SAMPLES is not None:
        samples = samples[:MAX_SAMPLES]
    patient_ids = [sample_id for _, _, sample_id in samples]
    print(f"{len(patient_ids)} patients, {len(ORGANS)} organs")
    print(f"using finetuned checkpoint: {FINETUNED_CHECKPOINT}")

    self_row_idx = build_self_row_idx(unique_reports, patient_ids, captions)

    model = build_eval_model(FINETUNED_CHECKPOINT, device)

    image_feats, image_valid = compute_image_embeddings(model, patient_ids, device)
    print(f"image embeddings computed for {image_valid.sum()}/{image_valid.size} (patient, organ) pairs")

    top_k_indices = compute_top_k_indices(image_feats, image_valid, bank_text_feats, self_row_idx, TOP_K, device)

    records = []
    for i, pid in enumerate(patient_ids):
        record = {"id": pid, "image_path": f"{pid}.nii.gz"}
        for organ_idx, organ in enumerate(ORGANS):
            record[organ] = clean_caption(captions[pid][organ], organ)
            entries = unique_reports[organ]
            # -1（没有可用 image embedding，或者 k < top_k 空出来的槎位）转成
            # null。其余的 idx 是 bank 里第几行去重文本，从那一行的 patient_ids
            # 里挑第一个当代表——一定不是查询病人自己（见模块开头的说明）。
            record[f"{organ}_indices"] = [
                {"id": entries[idx]["patient_ids"][0], "organ": organ} if idx >= 0 else None
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
