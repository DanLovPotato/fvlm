"""KE_prepare_unique 第 1 步：把 train 病人的报告文本按器官去重，为后面"去重版"知识库
（跟 KE_prepare 里那版"每个病人一条"的知识库不一样）做准备。

背景：KE_prepare 那版是每个 (病人, 器官) 都单独存一条文本、单独算一次 embedding——但
同一句话在很多病人之间是完全重复的，最极端的就是模板句 "{organ} shows no significant
abnormalities."，成千上万个病人共享同一句话。

这份脚本把"文本"和"哪些病人说了这句话" 拆开：每个器官下，相同的文本只保留一份，同时记下所有原本说过这句话的 patient_id

1. 调用 finetune.py 里的 build_organ_captions 读取原始报告，并用 CTOrganDataset 的过滤逻辑筛出图像、mask、caption 三者都存在的训练病人，保证病人集合跟后续真正算 embedding 时用的病人集合完全一致。
2. 对每个病人的每个器官，用跟 KE_prepare/compute_text_embeddings.py 里完全相同的 clean_caption() 清洗文本（这点很关键：必须用清洗后的文本去重，否则清洗前不同、清洗后相同的文本会被误判为不同）。
3. 按器官分组去重：同一器官下，相同文本的所有病人 ID 合并进同一条记录；不同器官之间即使文本恰好相同也不合并，因为下游检索是按器官分别计算相似度矩阵的。
4. 输出 unique_reports.json，格式是 {organ: [{"text": ..., "patient_ids": [...]}, ...]}。

输出：unique_reports.json，长这样：
    {
        "abdomen": [
            {"text": "abdomen shows no significant abnormalities.", "patient_ids": ["train_10000_a_1", ...]},
            {"text": "...真实所见...", "patient_ids": [...]},
            ...
        ],
        "bone": [...],
        ...  每个器官一个列表，列表里每一项是一条去重后的文本 + 说过这句话的所有病人
    }
"""
import json
import os
import sys

from tqdm import tqdm

_FVLM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _FVLM_ROOT)
# compute_text_embeddings.py 住在 KE_prepare/ 这个兄弟目录下，不在 fvlm/ 根目录，
# 单靠上面那行 sys.path 找不到它，要把 KE_prepare/ 也加进搜索路径。
sys.path.insert(0, os.path.join(_FVLM_ROOT, "KE_prepare"))

from finetune import CTOrganDataset, ORGANS, build_organ_captions
from compute_text_embeddings import clean_caption


DATA_ROOT = r"/mnt/researchdrive/ptiwari9/Staff_Trainee_Folders/Dan/chestCT/data/dataset"
RADGENOME_CSV = os.path.join(DATA_ROOT, "radgenome_files", "train_region_report.csv")
OUTPUT_DIR = os.path.join(DATA_ROOT, "EK_files_train_unique")
MAX_SAMPLES = None  # 改成一个数字可以只跑一小部分病人，用于冒烟测试


def get_unique_reports(patient_ids, captions):
    """按器官分组，返回 {organ: [{"text": str, "patient_ids": [str, ...]}, ...]}。

    同一个器官下，文本相同的病人被合并进同一条记录的 patient_ids 列表里；顺序保留
    "第一次见到这条文本时"的先后，方便复现（不是按出现频次排序 - 频次高低留给下游
    自己需要的时候再统计,这里只负责去重)。
    """
    unique_reports = {organ: {} for organ in ORGANS}  # organ -> {text: [patient_id, ...]}

    for pid in tqdm(patient_ids, desc="dedupe"):
        for organ in ORGANS:
            text = clean_caption(captions[pid][organ], organ)
            unique_reports[organ].setdefault(text, []).append(pid)

    return {
        organ: [
            {"text": text, "patient_ids": pids}
            for text, pids in text_to_pids.items()
        ]
        for organ, text_to_pids in unique_reports.items()
    }


def main():
    captions = build_organ_captions(RADGENOME_CSV, ORGANS)
    # 跟 compute_text_embeddings.py 用同一套 CTOrganDataset 过滤逻辑（要求 image、
    # mask、caption 三者同时存在），保证这里统计的病人集合和后面真正算 embedding 的
    # 病人集合完全一致。vis_processor 不会被用到（这里不调用 __getitem__），传 None。
    dataset = CTOrganDataset(ORGANS, captions, vis_processor=None)
    samples = dataset.samples
    if MAX_SAMPLES is not None:
        samples = samples[:MAX_SAMPLES]
    patient_ids = [sample_id for _, _, sample_id in samples]
    print(f"{len(patient_ids)} patients, {len(ORGANS)} organs")

    unique_reports = get_unique_reports(patient_ids, captions)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "unique_reports.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(unique_reports, f, ensure_ascii=False)
    print(f"Saved {out_path}")

    print("\nper-organ dedup stats (unique texts / total patients):")
    for organ in ORGANS:
        n_unique = len(unique_reports[organ])
        n_real = sum(1 for entry in unique_reports[organ] if entry["text"] != f"{organ} shows no significant abnormalities.")
        print(f"  {organ}: {n_unique}/{len(patient_ids)} unique texts ({n_real} of them non-template)")


if __name__ == "__main__":
    main()
