# 本地新增/修改内容说明

这份 repo fork 自原作者 jianpeng.zjp 的 fVLM 官方代码（原始 README 见 [README.md](README.md)，只发布了覆盖 4 个器官 lung/heart/esophagus/aorta 的预训练 checkpoint）。下面记录的是在此基础上**我自己新增/修改**的内容，分成两条主线：

1. **器官级微调主流程**：把原始 4 器官 checkpoint 扩展成 10 器官（新增 abdomen/bone/breast/mediastinum/pleura/thyroid/trachea and bronchie 这 7 个器官的 adapter，冻结原有 3 个器官的权重不动），并配套评估、画图脚本。
2. **知识增强（KE）检索流程**：给每个病人每个器官的报告文本建一个向量库，训练/推理时可以用病人自己的 CT 图像去检索"其他病人最相似的报告"作为参考信息。这条线做了两版，第二版（`KE_prepare_unique/`）是去重后的版本，是这几天在会话里跟 Claude 逐步一起写的。

代码里能看到的文件顶部注释已经写得很详细（中文，解释"为什么这么写"），这里只做一个导航，具体细节建议直接打开对应文件看注释。

---

## 1. 器官级微调主流程
(这些是我们自己写的)
| 文件 | 作用 |
|---|---|
| [preprocess.py](preprocess.py) | 离线预处理：把原始 CT 体数据按器官 mask 的包围盒裁剪、重采样、pad 到固定尺寸后存盘。逻辑核心跟原仓库 `data/preprocess.py` 一致，但前端改成读这批数据实际的目录结构（每个器官一个单独的 mask 文件，需要先合并成一张整数分割图）。`finetune.py` 训练时只读这一步的输出，不直接读原始体数据。 |
| [finetune.py](finetune.py) | 微调主脚本。把发布的 4 器官 checkpoint 的组织头（query_tokens/vision_proj）扩展到 10 个器官：重叠的 3 个器官（lung/heart/esophagus）保持预训练权重**冻结**，新增的 7 个器官随机初始化、参与训练。`pleura` 跟 `lung` 共用同一个 mask 分割值（数据里没有单独标注），但各自有独立的 query_tokens/vision_proj。 |
| [finetune.yaml](finetune.yaml) | `finetune.py` 用的训练配置（超参数等）。 |
| [eval_finetune.py](eval_finetune.py) | 针对微调后 10 器官 checkpoint 的零样本病理分类评估，从原仓库 `eval.py` 改写而来（没有直接改 `eval.py`，因为那是官方 4 器官版本的参考代码）。跟原版 `eval.py` 的三点区别：器官-id 映射方式、模型构建方式（要扩展到 10 器官）、评估用的 pathology 列表。 |
| [compare_retrieval_Rk.py](compare_retrieval_Rk.py) | 在验证集上对比"原始未微调 checkpoint" vs "微调后 checkpoint"的图文检索指标（R@1/R@5/R@10），按器官分别统计，包括原始 checkpoint 完全没见过的 6 个新器官（预期只有接近随机水平，作为微调效果的基线）。 |


## 2. 知识增强检索流程 v1（按病人建库，`KE_prepare/`）

给每个病人、每个器官的报告文本单独算一条 embedding，检索时排除自己（同一批病人既是 query 又是 bank，排除对角线）。

| 文件 | 作用 |
|---|---|
| [KE_prepare/compute_text_embeddings.py](KE_prepare/compute_text_embeddings.py) | 只搭模型的文本这一半（tokenizer + text_encoder + text_proj），给每个 (病人, 器官) 的报告文本算 embedding，存成 `organ_report_embeddings.npz`（稠密数组，`feats[病人下标, 器官下标]`）。 |
| [KE_prepare/find_similar_reports.py](KE_prepare/find_similar_reports.py) | 给每个 train 病人、每个器官算图像 embedding（要用微调后的 checkpoint），去上一步的文本库里检索最相似的 top-k 个**其他** train 病人的报告，排除自己（相似度矩阵对角线设 -inf）。输出 `organ_annotation.json`。 |
| [KE_prepare/find_similar_reports_val.py](KE_prepare/find_similar_reports_val.py) | 同上，但 query 是 validation 病人的图像，bank 固定用 train 的文本库（两拨病人不重叠，不用排除自己，矩阵是矩形的）。 |
| [KE_prepare/clean_null_indices.py](KE_prepare/clean_null_indices.py) | 清理 `organ_annotation.json` 里 `_indices` 列表中的 `null` 占位符（某些病人某个器官没有 mask、检索不出结果时会填 null，下游训练代码遇到非空列表里混了 None 会崩溃，这里过滤掉）。 |
| [KE_prepare/clean_null_indices_val.py](KE_prepare/clean_null_indices_val.py) | 同上，处理 validation 版本的 `organ_annotation.json`。 |

## 3. 知识增强检索流程 v2（去重建库，`KE_prepare_unique/`）

v1 的问题：同一个器官下，很多病人的报告文本一字不差（尤其是"未见明显异常"这种模板句），按病人建库会有大量重复 embedding，白白浪费计算。v2 先按文本内容去重，再建库，同一句话全库只算一次 embedding。这四个文件按运行顺序编号。

| 文件 | 作用 |
|---|---|
| [KE_prepare_unique/1_get_unique_report.py](KE_prepare_unique/1_get_unique_report.py) | 按器官把所有病人的报告文本分组去重，输出 `unique_reports.json`：`{器官: [{"text": 去重文本, "patient_ids": [说过这句话的所有病人]}, ...]}`。 |
| [KE_prepare_unique/2_compute_text_embeddings.py](KE_prepare_unique/2_compute_text_embeddings.py) | 给上一步每条去重后的文本算一次 embedding（不再是每个病人都算一次）。因为去重后每个器官剩下的文本条数差别很大，输出不再是统一的三维稠密数组，而是按器官分别存一个 `"{器官}_feats"` 二维数组，行号对应 `unique_reports.json` 里同一器官列表的下标。 |
| [KE_prepare_unique/3_find_similar_reports.py](KE_prepare_unique/3_find_similar_reports.py) | 给每个 train 病人、每个器官算图像 embedding，去上一步的去重文本库里检索最相似的 top-k。跟 v1 的区别：bank 不再跟病人一一对应，相似度矩阵是矩形的（病人数 × 该器官去重文本数），排除自己不再是排对角线，而是先查这个病人自己的文本在库里对应哪一行、再把那一行设成 -inf。输出格式跟 v1 的 `organ_annotation.json` 保持一致（方便下游代码复用）。 |
| [KE_prepare_unique/4_clean_null_indices.py](KE_prepare_unique/4_clean_null_indices.py) | 跟 v1 的 `clean_null_indices.py` 逻辑一样，清理 `organ_annotation.json` 里的 `null` 占位符；默认路径目前还指向 v1 的目录（`EK_files_train`），跑的时候需要用 `--input` 显式指向 `EK_files_train_unique/organ_annotation.json`。 |

> 关于 v2 设计上一个需要注意的取舍：排除自己那一行时，如果这行文本被成千上万个病人共享（比如某个器官的正常模板句），会导致所有共享这句话的病人查询时都拿不到"这条本该是最匹配的参考"。目前的实现是严格排除，还没有按 `patient_ids` 数量做阈值区分（讨论过这个点，还没改，见后续对话）。

## 4. 对原有文件做了实质性修改的（不是行尾格式那种噪音改动）

repo 里绝大部分 `lavis/` 下的文件虽然 `git diff` 显示改动很大，但实测只是行尾从 CRLF 转成了 LF（原始上传的文件是 Windows 换行），不是真实逻辑改动。真正有实质内容修改的是：

| 文件 | 改了什么 |
|---|---|
| [lavis/models/blip_models/blip_pretrain.py](lavis/models/blip_models/blip_pretrain.py) | 模型前向逻辑改动，支撑 10 器官扩展（`expand_organs`）、`forward_test_win` 检索用的图像 embedding 输出等，具体见文件内注释。 |
| [lavis/models/med.py](lavis/models/med.py) | `XBertEncoder` 相关改动（`set_output_embeddings` 补丁等），`KE_prepare*/compute_text_embeddings.py` 里 `_apply_environment_patches()` 依赖的就是这里的改动。 |
| [eval.py](eval.py) | 改了验证集图像目录的读取方式（从写死的相对路径 `data/processed_valid_images` 改成拼 `finetune.py` 里的 `DATA_ROOT`），并加了 `_apply_environment_patches()` 调用。 |


