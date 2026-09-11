"""KE_prepare_unique 第 2 步：给 1_get_unique_report.py 去重后的每一条文本算一次
embedding，存成 organ_report_embeddings.npz。

输入直接是 1_get_unique_report.py 产出的 unique_reports.json。

输出：organ_report_embeddings.npz 本身是二进制格式，不是 JSON；下面这段是把它当作
JSON 摆出来的示意（数值取真实跑出来的结果，256 维向量只截取前 6 维、用 "..." 省略
剩下的，行也只展示前 2 行），方便直观看懂里面是什么：

    {
        "organs": ["abdomen", "bone", "breast", ..., "pleura"],   // 10 个器官名
        "lung_feats": [
            [-0.0484, -0.0969, -0.0445, -0.0933, 0.0357, 0.0204, ...],   // 对应
            [-0.0480, -0.0972, -0.0427, -0.0927, 0.0353, 0.0222, ...],   // unique_reports
            ...                                                          // .json 里
        ],                                                               // lung[0], lung[1]...
        "breast_feats": [
            [-0.0467, -0.0982, -0.0439, -0.0933, 0.0351, 0.0213, ...],
            ...
        ],
        ...   每个器官一个 "{organ}_feats"，实际行数（去重后的唯一文本数）差别很大：
              lung 19955 行、abdomen 13371 行、... breast 831 行最少
    }

每个 "{organ}_feats" 的行数就是这个器官去重后剩下的唯一文本数，每一行都是 256 维、
已经 L2 归一化过（范数为 1）。第 i 行对应 unique_reports.json 里 data[organ][i]
（同一个器官内部两者顺序完全一致，见 main() 里的写法）——这份 npz 里故意不重复存
文本或 patient_ids，要查某条 embedding 对应哪条文本、哪些病人，去
unique_reports.json 按下标查即可，避免两份文件各存一份、以后改了一边忘了改另一边
导致对不上。
"""
import json
import os
import sys
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finetune import CHECKPOINT_PATH, ORGANS, _apply_environment_patches


DATA_ROOT = r"/mnt/researchdrive/ptiwari9/Staff_Trainee_Folders/Dan/chestCT/data/dataset"
UNIQUE_DIR = os.path.join(DATA_ROOT, "EK_files_train_unique")
UNIQUE_REPORTS_JSON = os.path.join(UNIQUE_DIR, "unique_reports.json")
OUTPUT_DIR = UNIQUE_DIR

# 这里用的是原生 4 器官的 CHECKPOINT_PATH，不是某个微调过的 checkpoint——因为
# text_encoder/text_proj 在发布的原始 checkpoint 和任何微调后的 checkpoint 之间都是
# 同一份冻结权重，完全一样，所以 expand_organs()、用哪个微调 checkpoint，这些跟
# "算文本 embedding"这件事毫无关系，改这个常量也不需要（跟 KE_prepare 那版同理）。
CHECKPOINT = CHECKPOINT_PATH
BATCH_SIZE = 64
MAX_SAMPLES = None  # 改成一个数字可以给每个器官只算前 N 条文本，用于冒烟测试


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

    # 打包成一个跟 compute_embeddings_for_texts() 期望的接口一致的轻量对象（只有属性
    # 访问，没有 nn.Module 的那套机制），这样下面 compute_embeddings_for_texts() 不用
    # 因为不再传完整的 BlipPretrain 模型而跟着改。
    return types.SimpleNamespace(
        tokenizer=tokenizer, text_encoder=text_encoder, text_proj=text_proj, max_txt_len=384,
    )


@torch.inference_mode()
def compute_embeddings_for_texts(model, texts, batch_size, device):
    """给一列已经清洗、去重过的文本逐批算 embedding，返回 (len(texts), 256) 的
    float32 数组，行顺序跟传入的 texts 顺序完全一致——不做任何按长度重排/分桶，
    调用方（main()）负责保证这个顺序跟 unique_reports.json 里对应器官的列表顺序
    对齐。
    """
    feats = np.zeros((len(texts), 256), dtype=np.float32)
    for start in tqdm(range(0, len(texts), batch_size), desc="embed"):
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
        feats[start:start + batch_size, :] = text_feat.cpu().numpy()
    return feats


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(UNIQUE_REPORTS_JSON, "r", encoding="utf-8") as f:
        unique_reports = json.load(f)
    assert set(unique_reports.keys()) == set(ORGANS), (
        f"unique_reports.json 里的器官跟当前 ORGANS 不一致: "
        f"json={sorted(unique_reports.keys())} vs ORGANS={sorted(ORGANS)}"
    )

    model = build_text_side(CHECKPOINT, device)

    save_dict = {"organs": np.array(ORGANS)}
    for organ in ORGANS:
        entries = unique_reports[organ]
        if MAX_SAMPLES is not None:
            entries = entries[:MAX_SAMPLES]
        # entries 的顺序就是 unique_reports.json 里 data[organ] 的原始顺序（1_get_unique_
        # report.py 按"第一次见到这条文本"的先后写进去的），这里原样取 text 不做任何
        # 排序/去重，保证第 i 行 embedding 对应 entries[i]，也就对应 json 文件里的第 i 条。
        texts = [entry["text"] for entry in entries]
        print(f"{organ}: {len(texts)} unique texts")
        feats = compute_embeddings_for_texts(model, texts, BATCH_SIZE, device)
        assert not np.isnan(feats).any(), f"{organ} 算出来的文本 embedding 里出现了 NaN"
        save_dict[f"{organ}_feats"] = feats

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    npz_path = os.path.join(OUTPUT_DIR, "organ_report_embeddings.npz")
    np.savez(npz_path, **save_dict)
    print(f"Saved {npz_path}")
    for organ in ORGANS:
        print(f"  {organ}_feats shape: {save_dict[f'{organ}_feats'].shape}")


if __name__ == "__main__":
    main()
