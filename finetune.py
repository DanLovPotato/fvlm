"""Fine-tune the fVLM organ-contrastive head on RadGenome-CT-RATE region masks/reports.

Lives inside fvlm_original (not a separate package) so it can import lavis.* the same
way train.py/eval.py already do.

The released checkpoint (model.pth) was pretrained over exactly 4 organs (lung, heart,
esophagus, aorta) - confirmed by the shape of its query_tokens/vision_projs. Our region
masks and RadGenome per-organ reports cover 10 organs, so this script expands the organ
head to all 10: the 3 that overlap with the checkpoint (lung, heart, esophagus) keep
their pretrained query_tokens row and vision_proj exactly as released - frozen, not just
warm-started - and the other 7 (abdomen, bone, breast, mediastinum, pleura, thyroid,
trachea and bronchie) get freshly initialized, trainable query_tokens rows and
vision_projs. No standalone aorta mask (any "mediastinum/aorta" RadGenome sentences fold
into the mediastinum caption via the "/"-split in build_organ_captions).

'pleura's raw mask is voxel-for-voxel identical to 'lung's in this dataset, so it's never
given its own id in the on-disk single-channel mask (MASK_SOURCE_ORGANS, still 9 organs,
written by preprocess.py) - ORGAN_MASK_ID instead points "pleura" at lung's id, so both
organs are read from the same seg value (see count_intact_organs /
blip_pretrain.py::forward()). Both still get their own query_tokens row, vision_proj,
and RadGenome caption.

Everything shared across organs - the visual encoder, text encoder, text_proj,
temperature, and the cross-attention pooling module - is also frozen, since training
any of it would shift the 3 old organs' outputs away from the released checkpoint.
Only the new organs' query_tokens rows and vision_projs train.

Model construction and checkpoint loading reuse the original classes/methods directly
(ViT, XBertEncoder, BlipPretrain, BaseModel.load_checkpoint, BaseModel.get_optimizer_params)
instead of reimplementing them - see build_model()/expand_organs() below. We build the
model at its native 4-organ shape first (so BaseModel.load_checkpoint's strict=False
load matches every tensor shape exactly, MLM head aside) and only expand to 10 organs
*after* the checkpoint is in, rather than pre-expanding and hand-splicing the state dict.
"""
import argparse
import os
import random
import sys
from collections import defaultdict

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import wandb
import yaml
from monai import transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler


DATA_ROOT = r"/mnt/researchdrive/ptiwari9/Staff_Trainee_Folders/Dan/chestCT/data/dataset"
# Written by preprocess.py - already cropped to organ-bbox+margin and padded to
# CROP_SIZE, exactly like fvlm_original's own processed_{split}_images/masks
# convention. Nothing here reads the raw train_preprocessed/train_region_mask
# volumes directly; that only happens in preprocess.py.
# PREPROCESSED_IMAGE_ROOT = os.path.join(DATA_ROOT, "processed_valid_images")
# PREPROCESSED_MASK_ROOT = os.path.join(DATA_ROOT, "processed_valid_masks")
PREPROCESSED_IMAGE_ROOT = os.path.join(DATA_ROOT, "processed_train_images")
PREPROCESSED_MASK_ROOT = os.path.join(DATA_ROOT, "processed_train_masks")
CHECKPOINT_PATH = r"/mnt/researchdrive/ptiwari9/Staff_Trainee_Folders/Dan/chestCT/weights/fvlm_weights/model.pth"
RADGENOME_CSV = os.path.join(DATA_ROOT, "radgenome_files", "train_region_report.csv")
# Where finetune.py actually writes checkpoints (main() reads this same output_dir key
# out of finetune.yaml at run time - see main()/args.output_dir). Read straight from the
# yaml instead of duplicating the path as a second hardcoded constant here, which is
# exactly what let this drift out of sync with the yaml before (eval_finetune.py/
# KE_prepare's *.py import DEFAULT_OUTPUT_DIR to find checkpoints, so a stale copy here
# silently pointed them at an old run's directory). Only tracks the default
# --cfg-path=finetune.yaml; a run launched with a different --cfg-path writes wherever
# that file's own output_dir says, independent of this constant.
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "finetune.yaml")) as _f:
    DEFAULT_OUTPUT_DIR = yaml.safe_load(_f)["output_dir"]

ORGANS = [
    "abdomen", "bone", "breast", "esophagus", "heart",
    "lung", "mediastinum", "thyroid", "trachea and bronchie", "pleura"
]

# On-disk single-channel mask (preprocess.py::merge_organ_masks, id =
# MASK_SOURCE_ORGANS.index(organ) + 1) only encodes these 9 - pleura's raw mask is
# identical to lung's, so it never got its own id there.
MASK_SOURCE_ORGANS = [o for o in ORGANS if o != "pleura"]

# organ -> which seg value to look for. Same as MASK_SOURCE_ORGANS.index(organ) + 1 for
# everyone except "pleura", which reads lung's value instead of getting its own.
ORGAN_MASK_ID = {
    organ: MASK_SOURCE_ORGANS.index("lung" if organ == "pleura" else organ) + 1
    for organ in ORGANS
}

# Organs whose query_tokens row and vision_proj come from the pretrained checkpoint and
# stay frozen. The checkpoint's 4th organ, aorta, has no standalone mask here (its
# RadGenome sentences fold into mediastinum instead), so it's dropped rather than kept
# as an unused 4th frozen slot.
FROZEN_ORGANS = ("esophagus", "heart", "lung")

CROP_SIZE = (112, 256, 352)   # (D, H, W) - matches the checkpoint's fixed 1232-token position embedding
PATCH_SIZE = (16, 16, 32)

def count_intact_organs(seg, organs):
    """Per sample in the batch, which organs actually survive the random crop intact.

    Mirrors BlipPretrain.forward()'s own organ_mask_flags computation
    (blip_models/blip_pretrain.py:130-150) exactly - kept as a separate copy here
    rather than having forward() return it, since blip_pretrain.py is reused
    unmodified from upstream (see module docstring). An organ only counts if its
    mask is present AND doesn't touch the crop's boundary (RandSpatialCropd can
    slice through the middle of an organ; a half-organ isn't a usable "case").

    Looks each organ's presence up by ORGAN_MASK_ID (not by position - "pleura" and
    "lung" share the same seg value) rather than assuming seg value == index + 1.

    Returns a (batch_size, len(organs)) bool tensor.
    """
    flags = torch.zeros(len(seg), len(organs), dtype=torch.bool, device=seg.device)
    for i, pul_seg in enumerate(seg):
        boundaries = [
            pul_seg[0], pul_seg[-1],
            pul_seg[:, 0], pul_seg[:, -1],
            pul_seg[:, :, 0], pul_seg[:, :, -1],
        ]
        non_zero_boundaries = [b[b != 0].flatten() for b in boundaries]
        boundary_ids = set(torch.unique(torch.cat(non_zero_boundaries)).tolist())
        present_ids = set(torch.unique(pul_seg).tolist()) - {0}
        intact_ids = present_ids - boundary_ids

        for organ_idx, organ in enumerate(organs):
            flags[i, organ_idx] = ORGAN_MASK_ID[organ] in intact_ids
    return flags


def _clean_text(text):
    for ch in ('"', "'", "(", ")"):
        text = text.replace(ch, "")
    return text.strip()


def build_organ_captions(csv_path, organs):
    """patient_id -> {organ: caption}, get RadGenome sentences per top-level organ."""
    # Step 1:读 CSV,丢掉没有 Anatomy 或 Sentence 的行——这些是整份报告的汇总行
    # (Anatomy 是空的),不属于任何一个具体器官,没法归类。
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["Anatomy", "Sentence"])

    # Step 2:从 Anatomy 取出顶层器官名。Anatomy 是层级路径(比如
    # "trachea and bronchie/trachea"、"mediastinum/aorta"),只取第一个 "/" 之前
    # 的部分,这样不管子解剖结构写得多深,都能正确归到顶层器官上。
    # 同时从 Volumename 去掉后缀,得到 patient_id,方便按病人分组。
    df["organ"] = df["Anatomy"].str.split("/").str[0].str.strip()
    df["patient_id"] = df["Volumename"].str.replace(".nii.gz", "", regex=False)

    # Step 3:按 (patient_id, organ) 分组收集句子。
    # - 只保留 Anatomy 精确等于某个目标器官名的行(row.Anatomy,不是 row.organ)——
    #   也就是只要顶层这一行,不要任何 "/" 子级。查过真实 CSV:顶层这一行本身就
    #   已经是这个器官完整的汇总段落,更深的子级行只是把同一段话拆成更细的小块
    #   (比如 "bone/bone/spinal canal" 的句子,本来就是 "bone" 那句话里的一个分
    #   句),再拼接进来只会让同一段临床内容重复出现。
    # - if sent not in ... 这个去重判断留着没删,但现在其实用不上了:改成精确匹配
    #   Anatomy 之后,每个 (patient_id, organ) 最多只有一行数据能命中上面的过滤
    #   条件,所以下面这个 list 里最多只会装一句话——没有第二句进来,自然也就
    #   没有"重复"可判断,这行代码留着不影响结果,只是不会再真正起作用了。
    sentences = defaultdict(lambda: defaultdict(list))
    for row in df.itertuples():
        if row.Anatomy not in organs:
            continue
        sent = str(row.Sentence).strip()
        if sent and sent not in sentences[row.patient_id][row.organ]:
            sentences[row.patient_id][row.organ].append(sent)

    # Step 4:为每个病人、每个目标器官生成最终 caption。
    # - 收集到句子的:把这个器官所有句子拼接起来,再清洗一下文本(去掉引号/括号)。
    # - 没收集到任何句子的(病人报告里压根没提到这个器官):填一句默认模板
    #   "{organ} shows no significant abnormalities.",跟"确认过无异常"区分开来
    #   靠的是 organ_abnormal_flags(在 CTOrganDataset.__getitem__ 里,判断caption
    #   是不是以这句模板开头)。
    captions = {}
    for patient_id, organ_sents in sentences.items():
        captions[patient_id] = {}
        for organ in organs:
            sents = organ_sents.get(organ)
            if sents:
                captions[patient_id][organ] = _clean_text(" ".join(sents))
            else:
                captions[patient_id][organ] = f"{organ} shows no significant abnormalities."
    return captions


class CTOrganDataset(Dataset):
    """Mirrors lavis/datasets/datasets/caption_datasets.py's CaptionDataset as closely
    as the different data source allows: same loader shape (LoadImaged only - no
    Transposed/ScaleIntensityRanged/cropping here, since preprocess.py already did
    that offline and saved the fixed-size result), same vis_processor
    (BlipImageTrainProcessor, passed in - the real registered class, not a
    reimplementation), same per-organ abnormal-flag logic in __getitem__, same
    retry-by-reindexing behavior on a bad sample. The only real difference is where
    captions come from: CaptionDataset reads desc_info.json/conc_info.json keyed by
    patient; here they come from build_organ_captions()'s RadGenome CSV parse, since
    that's the annotation source this dataset actually has.
    """
    def __init__(self, organs, captions, vis_processor):
        self.organs = organs
        self.captions = captions
        self.vis_processor = vis_processor
        self.loader = transforms.Compose([
            transforms.LoadImaged(keys=["image", "label"], image_only=True, ensure_channel_first=True),
        ])

        self.samples = []
        for file_name in sorted(os.listdir(PREPROCESSED_IMAGE_ROOT)):
            if not file_name.endswith(".nii.gz"):
                continue
            sample_id = file_name[:-len(".nii.gz")]
            label_path = os.path.join(PREPROCESSED_MASK_ROOT, file_name)
            if not os.path.exists(label_path) or sample_id not in captions:
                continue
            self.samples.append((os.path.join(PREPROCESSED_IMAGE_ROOT, file_name), label_path, sample_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # Matches CaptionDataset.__getitem__'s own retry loop (caption_datasets.py:
        # 96-129): on any load error, print and retry from a random other index
        # instead of raising, since a handful of bad files shouldn't kill a run.
         # __getitem__ 本身是被动的:给它哪个 index,它就老老实实加载、处理、返回这一个样本。
        # 内部唯一会自己"随机换一个"的地方,是下面失败重试那段——某个 index 加载失败时,
        # 才会随机挑个别的 index 再试一次,这只是错误兜底,不是常规取数据的方式。
        exit_ = False
        while not exit_:
            try:
                img_path, label_path, sample_id = self.samples[index]

                data = self.loader({"image": img_path, "label": label_path})
                data = self.vis_processor(data)
                image = data["image"].as_tensor()
                seg = data["label"][0].as_tensor()
                assert image[0].shape == CROP_SIZE and seg.shape == CROP_SIZE

                text_input = dict(self.captions[sample_id])
                organ_abnormal_flags = torch.zeros(len(self.organs), dtype=torch.bool)
                for organ_id, organ in enumerate(self.organs):
                    if organ in text_input and not text_input[organ].startswith(
                            f"{organ} shows no significant abnormalities."):
                        organ_abnormal_flags[organ_id] = True
                    if organ not in text_input:
                        text_input[organ] = f"{organ} shows no significant abnormalities."

                exit_ = True

            except Exception as e:
                print(e, self.samples[index][0])
                index = random.randint(0, len(self.samples) - 1)
                continue

        return {
            "image": image,
            "seg": seg,
            "text_input": text_input,
            "organ_abnormal_flags": organ_abnormal_flags,
        }

# 这个函数的作用是把这一堆"单个样本的 dict"拼成"一个 batch 的 dict":
# - image/seg/organ_abnormal_flags:用 torch.stack 沿新的第0维堆起来,变成
#   (batch_size, ...) 的 tensor,是最普通的 batch 化。
# - text_input 不能直接堆,因为它是字符串组成的 dict,不是 tensor。这里做的是
#   "转置":把"每个样本各自的 {器官: caption}"转成"每个器官对应一个长度为
#   batch_size 的 caption 列表"——正好是 BlipPretrain.forward() 需要的形状
def collate_fn(batch):
    organs = list(batch[0]["text_input"].keys())
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "seg": torch.stack([b["seg"] for b in batch]),
        "text_input": {organ: [b["text_input"][organ] for b in batch] for organ in organs},
        "organ_abnormal_flags": torch.stack([b["organ_abnormal_flags"] for b in batch]),
    }


def _apply_environment_patches():
    """Patches for gaps in *this* environment (missing local tokenizer/config assets,
    a newer transformers than blip.py's hard pin, a real bug in XBertEncoder) - not
    workarounds for anything about calling from_config() vs constructing by hand.
    Kept isolated here so build_model() below reads as plain reuse of the original
    classes.
    """
    """这里打的三个补丁,都是为了填平"这台机器上的运行环境"跟原版代码预期环境之间
    的缝隙,不是在改模型本身的逻辑,跟"调用 from_config() 还是自己手动构造模型"这
    种设计选择没关系。特意单独抽成一个函数、放在这里调用,是为了让下面 build_model()
    读起来就是"纯粹复用原版的类",不被这些补丁细节干扰。
    """
    from transformers import BertTokenizer

    from lavis.models.blip_models import blip as blip_module
    from lavis.models.med import XBertEncoder

    # blip.py asserts transformers<4.27, inherited from upstream BLIP's reliance on
    # HF's BertModel internals. med.py is a from-scratch reimplementation of BERT
    # (only using stable base classes), so that constraint doesn't apply here.
    blip_module.BlipBase.__init__ = nn.Module.__init__

    @classmethod
    def _init_tokenizer(cls):
        return BertTokenizer.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")

    blip_module.BlipBase.init_tokenizer = _init_tokenizer

    # XBertEncoder overrides get_output_embeddings() but never defines the matching
    # set_output_embeddings(), which newer transformers' PreTrainedModel requires when
    # tie_word_embeddings=False (resize_token_embeddings() calls it directly, and
    # there's no base-class fallback). The MLM head is unused for organ-contrastive
    # training, so this only needs to keep resize functional.
    def _set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    XBertEncoder.set_output_embeddings = _set_output_embeddings


def build_pretrained_model(checkpoint_path):
    """Construct BlipPretrain at its native 4-organ shape and load the released
    checkpoint into it via BaseModel.load_checkpoint() (base_model.py:29) - reused
    unmodified, since at 4 organs every tensor shape matches the checkpoint exactly
    and load_checkpoint's strict=False already tolerates the (unused) missing MLM
    head keys. No manual state-dict surgery needed.
    """
    from transformers import BertConfig

    from lavis.models.blip_models.blip_pretrain import BlipPretrain
    from lavis.models.blip_models.vit import ViT
    from lavis.models.med import XBertEncoder

    # Matches microsoft/BiomedVLP-CXR-BERT-specialized's config, confirmed against the
    # checkpoint's text_encoder.* tensor shapes (30522 vocab, 12 layers, hidden 768).
    # Built by hand rather than read from med_config_path's json file, since that file
    # isn't present in this checkout and every weight it would seed gets overwritten
    # by the checkpoint load below anyway.
    bert_config = BertConfig(
        vocab_size=30522, hidden_size=768, num_hidden_layers=12,
        num_attention_heads=12, intermediate_size=3072,
        max_position_embeddings=512, type_vocab_size=2,
        hidden_dropout_prob=0.25, attention_probs_dropout_prob=0.25,
        layer_norm_eps=1e-12, pad_token_id=0,
        add_type_embeddings=False,
        tie_word_embeddings=False,
    )
    text_encoder = XBertEncoder(config=bert_config, add_pooling_layer=False)

    # No MAE-checkpoint init here (unlike BlipPretrain.from_config's pretraining path):
    # every visual_encoder.* weight it would seed is about to be overwritten by the
    # fVLM checkpoint below, so loading the MAE weights first would be pure waste.
    image_encoder = ViT(
        in_channels=1, img_size=CROP_SIZE, patch_size=PATCH_SIZE,
        num_classes=0, dropout_rate=0.1, qkv_bias=True,
    )

    model = BlipPretrain(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        text_decoder=None,
        embed_dim=256,
        alpha=0.4,
        tie_enc_dec_weights=False,
        max_txt_len=384,
    )

    msg = model.load_checkpoint(checkpoint_path)
    unexpected = [k for k in msg.unexpected_keys if k not in ("image_queue", "text_queue", "queue_ptr")]
    expected_missing = {
        f"text_encoder.cls.predictions.{k}" for k in (
            "bias", "transform.dense.weight", "transform.dense.bias",
            "transform.LayerNorm.weight", "transform.LayerNorm.bias",
        )
    }
    missing = [k for k in msg.missing_keys if k not in expected_missing]
    assert not unexpected, f"unexpected checkpoint keys: {unexpected}"
    assert not missing, f"missing checkpoint keys: {missing}"

    return model


def expand_organs(model, new_organs, frozen_organs, organ_mask_ids=None):
    """
    !!!froze MLP layer

    Swap model.organs (currently the checkpoint's original 4) for new_organs.
    Any organ present in both the old list and frozen_organs keeps its pretrained
    query_tokens row and vision_proj, frozen; every other organ gets a freshly
    initialized, trainable row/proj.

    query_tokens is one Parameter spanning all organs, so per-row freezing can't be
    done via requires_grad. Returns the new-index positions that are frozen; the
    caller registers a gradient-masking hook and a zero-weight-decay group for them
    (see build_model/build_optimizer) - done there rather than here so the hook
    attaches after model.to(device), since Module._apply can swap in a brand-new
    Parameter object during .to() and silently drop a hook registered beforehand.

    organ_mask_ids: optional {organ_name: seg_value} dict (see ORGAN_MASK_ID), stored on
    the model as model.organ_mask_ids (list aligned with model.organs) for
    BlipPretrain.forward() to read each organ's mask by value instead of by position -
    needed because "pleura" and "lung" share the same seg value. Defaults to the plain
    index+1 scheme (no aliasing) for callers that don't pass it (forward_test_win() never
    consults model.organ_mask_ids, so eval-only callers can leave this out).
    """
    # Step 1:先把旧的东西(4 个器官的名字、query_tokens、vision_projs)存起来,
    # 马上就要被覆盖掉了,后面复制权重的时候还要用。
    old_organs = model.organs
    old_query_tokens = model.query_tokens.data
    old_vision_projs = model.vision_projs
    vision_width = old_query_tokens.shape[1]
    embed_dim = old_vision_projs[0].out_features

    # Step 2:换成新的、10 个器官的形状——query_tokens 先全部清零,vision_projs
    # 全部随机初始化(nn.Linear 默认初始化),后面再挑几个器官把权重覆盖回去。
    model.organs = list(new_organs)
    model.organ_mask_ids = (
        [i + 1 for i in range(len(new_organs))] if organ_mask_ids is None
        else [organ_mask_ids[o] for o in new_organs]
    )
    model.query_tokens = nn.Parameter(torch.zeros(len(new_organs), vision_width))
    model.vision_projs = nn.ModuleList([nn.Linear(vision_width, embed_dim) for _ in new_organs])

    # Step 3:遍历旧的 4 个器官,把"新列表里还有、而且要冻结"的那几个
    # (lung/heart/esophagus)的旧权重,复制到它们在新列表里的新位置上,
    # 并把这几个 vision_proj 的参数冻结(requires_grad=False)。
    # 记录下这些新位置的下标(frozen_rows),给调用方用来处理 query_tokens
    # 的梯度屏蔽(因为 query_tokens 是共享 tensor,vision_proj 可以直接冻结,
    # 但 query_tokens 这一行没法这样冻)。
    frozen_rows = []
    for old_idx, organ in enumerate(old_organs):
        if organ not in new_organs or organ not in frozen_organs:
            continue
        new_idx = new_organs.index(organ)
        model.query_tokens.data[new_idx] = old_query_tokens[old_idx]
        model.vision_projs[new_idx].load_state_dict(old_vision_projs[old_idx].state_dict())
        for p in model.vision_projs[new_idx].parameters():
            p.requires_grad = False
        frozen_rows.append(new_idx)

    # Step 4:打印一下,哪些器官是直接沿用预训练权重、冻结不训练的,
    # 哪些是新加的、需要训练的——方便肉眼确认扩展结果对不对。
    print(f"Frozen (pretrained, unchanged) organ heads: {[new_organs[i] for i in frozen_rows]}; "
          f"trainable adapters: {[o for i, o in enumerate(new_organs) if i not in frozen_rows]}")

    return frozen_rows


def freeze_shared_modules(model):
    """
    !!!froze vit, bert, attention layer,
    The query_tokens rows for the 3 old organs get protected 
     by the gradient-masking hook in build_model()       frozen_row_mask

    freezes everything the model shares across all 10 organs, 
    so the 3 old organs' (lung/heart/esophagus) outputs stay bit-for-bit identical 
    to the released checkpoint — only expand_organs' new query_tokens rows/vision_projs 
    for the 7 new organs are meant to actually learn anything.

    """
    for module in (model.visual_encoder, model.text_encoder, model.attention, model.text_proj):
        for p in module.parameters():
            p.requires_grad = False
    model.temp.requires_grad = False
    model.visual_encoder.eval()
    model.text_encoder.eval()
    model.attention.eval()


def build_model(checkpoint_path, organs, frozen_organs, device, organ_mask_ids=None):
    _apply_environment_patches()
    model = build_pretrained_model(checkpoint_path)
    frozen_rows = expand_organs(model, organs, frozen_organs, organ_mask_ids)
    freeze_shared_modules(model)

    model = model.to(device)

    # Registered after .to(device) (see expand_organs' docstring for why): zeroes the
    # frozen organs' query_tokens gradient every backward pass so training the new
    # organs' rows can't leak into the ones copied from the released checkpoint.
    frozen_row_mask = torch.zeros(len(organs), 1, dtype=torch.bool, device=device)
    frozen_row_mask[frozen_rows] = True
    model.query_tokens.register_hook(lambda grad: grad.masked_fill(frozen_row_mask, 0.0))

    return model


def build_optimizer(model, lr, weight_decay):
    """Reuses BaseModel.get_optimizer_params (base_model.py:108) for the frozen/decay
    vs no-decay split instead of hand-building param groups - it already skips frozen
    params and puts 1-D params (bias/LayerNorm/etc.) in a no-decay group. The one thing
    it can't know is that query_tokens (2-D, so it lands in the decayed group) has
    frozen rows living inside an otherwise-trainable Parameter: those rows' gradient is
    already zeroed by the hook in expand_organs, but decoupled AdamW weight decay is
    applied independent of the gradient and would still erode them every step. So pull
    query_tokens out into its own zero-decay group after the fact.
    """
    # Plain `in`/`.remove()` would fall back to tensor __eq__ for any non-identical
    # element in the list (mismatched shapes -> RuntimeError, matched shapes -> an
    # ambiguous multi-element bool) - so filter by object identity instead.
    optim_params = model.get_optimizer_params(weight_decay=weight_decay)
    for group in optim_params:
        if any(p is model.query_tokens for p in group["params"]):
            group["params"] = [p for p in group["params"] if p is not model.query_tokens]
            break
    optim_params.append({"params": [model.query_tokens], "weight_decay": 0.0})
    return torch.optim.AdamW(optim_params, lr=lr)


def _relaunch_under_torchrun(cfg_path, gpus):
    """`python3 finetune.py` alone should be enough - no torchrun/CUDA_VISIBLE_DEVICES
    command to remember. If this process wasn't itself launched by torchrun (no
    LOCAL_RANK env var) and finetune.yaml names GPUs to use, re-exec this same script
    under `torchrun --nproc_per_node=len(gpus)` with CUDA_VISIBLE_DEVICES restricted to
    those GPUs, so init_distributed_mode() (called after this returns) finds the env
    vars it expects. os.execvpe replaces this process outright - torchrun then becomes
    the parent and spawns one real finetune.py process per GPU.
    """
    if "LOCAL_RANK" in os.environ or not gpus:
        return

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--nproc_per_node", str(len(gpus)),
        os.path.abspath(__file__), "--cfg-path", cfg_path,
    ]
    print(f"Relaunching under torchrun on GPUs {gpus}: {' '.join(cmd)}")
    os.execvpe(cmd[0], cmd, env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg-path", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "finetune.yaml"),
        help="YAML file with all fine-tuning settings, see finetune.yaml",
    )
    cli_args = parser.parse_args()

    with open(cli_args.cfg_path) as f:
        cfg = yaml.safe_load(f)

    _relaunch_under_torchrun(cli_args.cfg_path, cfg.pop("gpus", None))

    args = argparse.Namespace(**cfg)

    from lavis.common.dist_utils import get_rank, init_distributed_mode, is_main_process
    from lavis.common.optims import LinearWarmupCosineLRScheduler
    from lavis.processors.blip_processors import BlipImageTrainProcessor

    # Same launcher contract as train.py: torchrun sets RANK/WORLD_SIZE/LOCAL_RANK,
    # init_distributed_mode reads them and sets up the NCCL process group (falls back
    # to single-process/non-distributed if launched with plain `python`).
    init_distributed_mode(args)
    device = torch.device(args.gpu) if args.distributed else ("cuda" if torch.cuda.is_available() else "cpu")

    seed = args.seed + get_rank()
    random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    # Only rank 0 talks to W&B - every rank calling wandb.init() would create
    # len(gpus) duplicate runs.
    if is_main_process():
        wandb.init(
            project=args.wandb_project, name=args.wandb_run_name,
            mode=args.wandb_mode, config=vars(args),
        )

    captions = build_organ_captions(RADGENOME_CSV, ORGANS)
    # The actual registered "blip_image_train" processor (lavis/processors/
    # blip_processors.py:100-119) - does RandSpatialCropd(roi_size=CROP_SIZE) plus
    # random flips, exactly what CaptionDataset's own vis_processor does. Only
    # possible because preprocess.py already cropped/padded around the organ masks
    # offline, the same way fvlm_original's own data/preprocess.py does for its
    # 4-organ data - a spatially-blind crop wouldn't have been safe otherwise.
    vis_processor = BlipImageTrainProcessor()
    dataset = CTOrganDataset(ORGANS, captions, vis_processor)
    if args.max_samples is not None:
        dataset.samples = dataset.samples[:args.max_samples]
    print(f"Fine-tuning on {len(dataset)} samples, organs={ORGANS}")

    sampler = DistributedSampler(dataset, shuffle=True) if args.distributed else None
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler,
        drop_last=True, num_workers=args.num_workers, collate_fn=collate_fn,
    )

    model = build_model(args.checkpoint, ORGANS, FROZEN_ORGANS, device, ORGAN_MASK_ID)
    model.train()
    model.visual_encoder.eval()
    model.text_encoder.eval()
    model.attention.eval()

    optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = LinearWarmupCosineLRScheduler(
        optimizer, max_epoch=args.epochs, min_lr=args.min_lr, init_lr=args.lr,
        warmup_steps=args.warmup_steps, warmup_start_lr=args.warmup_lr)

    # find_unused_parameters=True: matches runner_base.py's own DDP wrapping - only the
    # vision_projs rows for organs actually present in a given batch get used, so which
    # trainable params participate varies step to step.
    if args.distributed:
        model = DDP(model, device_ids=[args.gpu], find_unused_parameters=True)
    model_without_ddp = model.module if args.distributed else model

    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)

    # Cumulative, across the whole run: how many (sample, organ) cases actually had
    # that organ intact after the random crop - see count_intact_organs() docstring
    # for why "intact" (not just "present"). This is independent of whether the
    # organ ended up with a usable contrastive target that step (organ_wise_loss_itm
    # below is stricter still, since it also requires an abnormal-caption sample).
    organ_case_counts = defaultdict(int)

    step = 0
    for epoch in range(args.epochs):
        if args.distributed:
            sampler.set_epoch(epoch)
        for i, batch in enumerate(loader):
            scheduler.step(cur_epoch=epoch, cur_step=i)

            batch["image"] = batch["image"].to(device)
            batch["seg"] = batch["seg"].to(device)
            batch["organ_abnormal_flags"] = batch["organ_abnormal_flags"].to(device)

            intact = count_intact_organs(batch["seg"], ORGANS)
            for organ_id, organ in enumerate(ORGANS):
                organ_case_counts[organ] += intact[:, organ_id].sum().item()

            output = model(batch)

            # ITC needs >=2 semantically-distinct samples per organ group to form a
            # contrastive target; when no organ in the batch clears that bar (small
            # batch, or all captions coincidentally identical/normal),
            # organ_wise_loss_itm is {} and output.loss is a plain python int 0, not
            # a tensor - nothing to backprop, so just skip the step.
            if not output.organ_wise_loss_itm:
                if step % args.log_freq == 0:
                    print(f"epoch {epoch} step {i}/{len(loader)}: no organ had a valid ITC target, skipped")
                step += 1
                continue

            loss = output.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % args.log_freq == 0:
                organ_losses = {k: round(v.item(), 4) for k, v in output.organ_wise_loss_itm.items()}
                lr = optimizer.param_groups[0]["lr"]
                print(f"epoch {epoch} step {i}/{len(loader)} lr {lr:.2e} loss {loss.item():.4f} {organ_losses}")
                if is_main_process():
                    wandb.log(
                        {"epoch": epoch, "lr": lr, "loss": loss.item(),
                         **{f"loss/{k}": v for k, v in organ_losses.items()}},
                        step=step,
                    )
            step += 1

        # organ_case_counts is only this rank's shard until summed here, so the printed
        # totals match what single-GPU would have reported.
        if args.distributed:
            counts = torch.tensor([organ_case_counts[o] for o in ORGANS], device=device)
            torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)
            organ_case_counts_world = dict(zip(ORGANS, counts.tolist()))
        else:
            organ_case_counts_world = dict(organ_case_counts)

        if is_main_process():
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_{epoch:03d}.pth")
            torch.save({"model": model_without_ddp.state_dict(), "epoch": epoch, "organs": ORGANS}, ckpt_path)
            print(f"Saved {ckpt_path}")
            print(f"  cumulative intact-organ cases through epoch {epoch}: {organ_case_counts_world}")
            wandb.log(
                {"epoch": epoch, **{f"intact_cases/{k}": v for k, v in organ_case_counts_world.items()}},
                step=step,
            )

    if is_main_process():
        wandb.finish()


if __name__ == "__main__":
    main()
