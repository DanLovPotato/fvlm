"""MvKeTR 知识库第 1/2 步：算每个 (病人, 器官) 的报告文本 embedding，存成
organ_report_embeddings.npz。


输出的 npz 是稠密数组，按 (病人, 器官) 两个下标定位：
    feats       (N, len(ORGANS), 256) float32   —— 文本 embedding（L2 归一化）
    patient_ids (N,)                  str       —— 第 i 个是哪个病人，对应 feats[i]
    organs      (len(ORGANS),)        str       —— 第 j 个是哪个器官，对应 feats[:, j]
查某个病人某个器官的向量：先用 patient_ids.index(pid) 找到 i，用 organs.index(organ)
找到 j，再取 feats[i, j]。
"""
import os
import sys
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import finetune
from finetune import (
     CTOrganDataset, CHECKPOINT_PATH, ORGANS,
    _apply_environment_patches, build_organ_captions,
)

DATA_ROOT = r"/mnt/researchdrive/ptiwari9/Staff_Trainee_Folders/Dan/chestCT/data/dataset"

# 改这一个变量就能切换 train/valid，不用再去手动改 finetune.py 里的注释。
SPLIT = "train"  # "train" 或 "valid"

# CTOrganDataset.__init__（finetune.py）内部直接读 finetune 模块自己的
# PREPROCESSED_IMAGE_ROOT/PREPROCESSED_MASK_ROOT 全局变量，不是构造函数参数——
# 所以不能只在这个文件里定义同名变量（那样 CTOrganDataset 根本看不到，还是会
# 用 finetune.py 里当前生效的那个）。必须真的去改 finetune 模块自己的属性，
# CTOrganDataset 是在调用时才查这两个名字，改了之后它就会用新值。
finetune.PREPROCESSED_IMAGE_ROOT = os.path.join(DATA_ROOT, f"processed_{SPLIT}_images")
finetune.PREPROCESSED_MASK_ROOT = os.path.join(DATA_ROOT, f"processed_{SPLIT}_masks")

# RADGENOME_CSV 不一样：build_organ_captions(csv_path, organs) 是当参数传进去的，
# 所以这里直接定义一个新的、不 import finetune.py 里那份就行。
_RADGENOME_CSV_NAME = {"train": "train_region_report.csv", "valid": "validation_region_report.csv"}[SPLIT]
RADGENOME_CSV = os.path.join(DATA_ROOT, "radgenome_files", _RADGENOME_CSV_NAME)

# 这里用的是原生 4 器官的 CHECKPOINT_PATH，不是某个微调过的 checkpoint——因为
# text_encoder/text_proj 在发布的原始 checkpoint 和任何微调后的 checkpoint 之间都是
# 同一份冻结权重，完全一样，所以 expand_organs()、用哪个微调 checkpoint，这些跟
# "算文本 embedding"这件事毫无关系，改这个常量也不需要。
CHECKPOINT = CHECKPOINT_PATH
# EK_files_train / EK_files_val - 跟 find_similar_reports.py/_val.py 用的目录名保持
# 一致（那两个脚本固定叫 "val" 不是 "valid"），所以这里单独映射一下后缀，不能直接拼
# f"EK_files_{SPLIT}"。
_EK_DIR_SUFFIX = {"train": "train", "valid": "val"}[SPLIT]
OUTPUT_DIR = os.path.join(DATA_ROOT, f"EK_files_{_EK_DIR_SUFFIX}")  # organ_report_embeddings.npz 存这里
BATCH_SIZE = 64
MAX_SAMPLES = None  # 改成一个数字可以只跑一小部分病人，用于冒烟测试


def build_text_side(checkpoint_path, device):
    """只搭文本encoder这一半（tokenizer + text_encoder + text_proj），不碰 ViT/vision_projs
    等图像部分。
    建 XBertEncoder + text_proj，从 checkpoint 里只挑
    "text_encoder."/"text_proj." 前缀的权重加载——跟 build_pretrained_model() 加载出来
    的是同一份权重，只是不用连 ViT 一起构造/加载。
    """
    from transformers import BertConfig, BertTokenizer
    from lavis.models.med import XBertEncoder

    _apply_environment_patches()  # 打上 XBertEncoder.set_output_embeddings 那个补丁，
                                   # resize_token_embeddings() 下面要用到

    tokenizer = BertTokenizer.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")

    # 跟 finetune.py 的 build_pretrained_model() 里手搭的 bert_config 完全一致。
    bert_config = BertConfig(
        vocab_size=30522, hidden_size=768, num_hidden_layers=12,
        num_attention_heads=12, intermediate_size=3072,
        max_position_embeddings=512, type_vocab_size=2,
        hidden_dropout_prob=0.25, attention_probs_dropout_prob=0.25,
        layer_norm_eps=1e-12, pad_token_id=0,
        add_type_embeddings=False, tie_word_embeddings=False,
    )
    text_encoder = XBertEncoder(config=bert_config, add_pooling_layer=False)
    text_encoder.resize_token_embeddings(len(tokenizer))
    text_proj = nn.Linear(768, 256)  # embed_dim=256，跟 BlipPretrain 的 text_proj 一致

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    text_encoder_sd = {
        k[len("text_encoder."):]: v for k, v in state_dict.items() if k.startswith("text_encoder.")
    }
    text_proj_sd = {
        k[len("text_proj."):]: v for k, v in state_dict.items() if k.startswith("text_proj.")
    }

    missing, unexpected = text_encoder.load_state_dict(text_encoder_sd, strict=False)
    # 跟 build_pretrained_model() 里同一份"预期缺失"名单，只是这里已经去掉了
    # "text_encoder." 前缀（因为是直接 load 进 text_encoder 自己，不是整个 BlipPretrain）。
    expected_missing = {
        f"cls.predictions.{k}" for k in (
            "bias", "transform.dense.weight", "transform.dense.bias",
            "transform.LayerNorm.weight", "transform.LayerNorm.bias",
        )
    }
    missing = [k for k in missing if k not in expected_missing]
    assert not unexpected, f"text_encoder 里出现了没预料到的 checkpoint key: {unexpected}"
    assert not missing, f"text_encoder 缺了本该有的 checkpoint key: {missing}"
    text_proj.load_state_dict(text_proj_sd, strict=True)

    text_encoder = text_encoder.to(device).eval()
    text_proj = text_proj.to(device).eval()

    # 打包成一个跟 compute_text_embeddings() 期望的接口一致的轻量对象（只有属性
    # 访问，没有 nn.Module 的那套机制），这样下面 compute_text_embeddings() 不用
    # 因为不再传完整的 BlipPretrain 模型而跟着改。
    return types.SimpleNamespace(
        tokenizer=tokenizer, text_encoder=text_encoder, text_proj=text_proj, max_txt_len=384,
    )


def clean_caption(text, organ):
    """跟 BlipPretrain.forward() 里对 caption 的清洗逻辑（blip_pretrain.py:220-224）
    完全一致，这样同一段 caption 文本，这里算出来的 embedding 才会跟训练/评估时用的
    是同一个东西——如果两边清洗方式不一样，文本 embedding 就对不上训练时模型真正见过
    的输入。

    只删"模板句 + 真实发现"这种拼接情况下多余的模板前缀，不删纯粹的模板句本身
    但其实在旧的 9 器官跑法上，24116 个病人 × 9 个器官没有一条命中这里的 if——
    加了 pleura（第 10 个器官）之后这个统计没有重新跑过，不能直接当结论用。
    """
    template = f"{organ} shows no significant abnormalities."
    if text.startswith(template) and text != template:
        # 真实所见 + 模板句子拼在一起的情况（比如 "abdomen shows no significant
        # abnormalities. 另外还发现了..."）：只有这种"以模板开头、但不完全等于模板"
        # 的才需要把模板部分去掉，只保留真正的发现内容。
        return text.replace(template, "")
    # 要么是纯模板句（没有真实发现），要么根本不是以模板开头，两种情况都原样返回。
    return text


@torch.inference_mode()
def compute_text_embeddings(model, patient_ids, captions, batch_size, device):
    """按器官为外层循环、病人为内层批次，逐批算出每个病人在每个器官下的文本
    embedding，写进一个 (N, len(ORGANS), 256) 的稠密数组里，直接就是最终要存盘
    的格式，不需要再额外处理。
    """
    feats = np.zeros((len(patient_ids), len(ORGANS), 256), dtype=np.float32)
    for organ_idx, organ in enumerate(ORGANS):
        # 同一个器官下，每个病人的 caption 长度差异不大（要么是真实所见，要么是同一句
        # 模板），所以按器官分组、统一 padding 到 max_txt_len，比按病人交叉着算更省
        # 显存也更好并行。
        texts = [clean_caption(captions[pid][organ], organ) for pid in patient_ids]
        for start in tqdm(range(0, len(texts), batch_size), desc=f"embed {organ}"):
            batch_texts = texts[start:start + batch_size]
            tokens = model.tokenizer(
                batch_texts, padding="max_length", truncation=True,
                max_length=model.max_txt_len, return_tensors="pt",
            ).to(device)
            text_output = model.text_encoder.forward_text(tokens)
            # 跟 BlipPretrain.forward() 一样，只取 [CLS] 位置（第 0 个 token）的
            # 隐层输出去投影，而不是对整句取平均——这是模型训练时用来做对比学习的
            # 那个句子级向量。
            text_embeds = text_output.last_hidden_state
            text_feat = F.normalize(model.text_proj(text_embeds[:, 0, :]), dim=-1)
            feats[start:start + batch_size, organ_idx, :] = text_feat.cpu().numpy()
    return feats


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    captions = build_organ_captions(RADGENOME_CSV, ORGANS)
    # 直接复用 CTOrganDataset 自己的文件过滤逻辑（要求 image、mask、caption 三者
    # 同时存在），这样这里用到的病人列表，跟 finetune.py 训练时实际用的病人列表、
    # 以及 find_similar_reports.py 之后要算图像 embedding 的病人列表完全一致，
    # 不会出现"文本这边有个病人，图像那边却没有"的错位。vis_processor 只有
    # __getitem__ 会用到，这里根本不调用 __getitem__，所以传 None 也没问题。
    dataset = CTOrganDataset(ORGANS, captions, vis_processor=None)
    samples = dataset.samples
    if MAX_SAMPLES is not None:
        samples = samples[:MAX_SAMPLES]
    # patient_ids = train_{病人号}_{scan}_{reconstruction}
    patient_ids = [sample_id for _, _, sample_id in samples]
    print(f"{len(patient_ids)} patients, {len(ORGANS)} organs")

    model = build_text_side(CHECKPOINT, device)

    feats = compute_text_embeddings(model, patient_ids, captions, BATCH_SIZE, device)
    assert not np.isnan(feats).any(), "算出来的文本 embedding 里出现了 NaN"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    npz_path = os.path.join(OUTPUT_DIR, "organ_report_embeddings.npz")
    np.savez(npz_path, feats=feats, patient_ids=np.array(patient_ids), organs=np.array(ORGANS))
    print(f"Saved {npz_path}: feats shape {feats.shape}")

    # 打印每个器官"真实所见"（非模板）样本占比：这个数字直接决定了这个器官在
    # find_similar_reports.py 检索时的知识库质量——如果某个器官大部分病人都只有
    # 模板句（没有真实病灶描述），检索出来的"相似报告"参考价值也会很有限，值得
    # 在正式用这份 npz 之前先看一眼这个统计。
    print("\nper-organ non-template (real finding) sample counts:")
    for organ in ORGANS:
        template = f"{organ} shows no significant abnormalities."
        n_real = sum(1 for pid in patient_ids if captions[pid][organ] != template)
        print(f"  {organ}: {n_real}/{len(patient_ids)}")


if __name__ == "__main__":
    main()
