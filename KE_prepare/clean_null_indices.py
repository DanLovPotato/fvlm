"""清理 organ_annotation.json 里 `_indices` 列表中的 null 占位符。

背景：find_similar_reports.py 生成 `<organ>_indices` 时，如果某个 (病人, 器官)
没有有效的 image embedding（这个器官的 mask 体素数为 0，见该脚本
compute_image_embeddings 里的 `if whole_organ_sizes[organ] == 0: continue`），
会把这一整行 5 个位置都填成 null（该脚本 main() 里 `... if idx >= 0 else None`）。

下游训练代码（src/Model/my_embedding_layer.py 的 _lookup_annotation_vectors）
在遍历 `<organ>_indices` 时，对"整个列表为空 []"这种情况有兜底处理，但没有对
"列表非空、但里面的某一项本身是 None"这种情况做判断，会直接
`neighbor["id"]` 崩溃（TypeError: 'NoneType' object is not subscriptable）。

这个脚本把每个 `_indices` 列表里的 None 过滤掉（不改变原本非 None 的
内容/顺序），让数据符合下游代码本来就支持的"列表可以比 top_k 短，
甚至可以是空列表"这条路径。实测（24116 个病人）显示同一个 `_indices`
列表要么 5 个位置全部有效、要么全部是 None，没有部分有效部分 None 的
混合情况——所以过滤后一个列表只会变短或者变空，不会出现"删了几个、
剩下几个真实近邻"这种情况。

用法：
    python clean_null_indices.py                 # 清理默认路径，覆盖原文件
    python clean_null_indices.py --dry-run        # 只打印统计，不写文件
    python clean_null_indices.py --input PATH --output PATH   # 自定义路径
"""
import argparse
import json
import os

DEFAULT_PATH = "/mnt/researchdrive/ptiwari9/Staff_Trainee_Folders/Dan/chestCT/data/dataset/EK_files_train/organ_annotation.json"


def clean_null_indices(data):
    """原地过滤 data['train'] 里每条记录的 `<organ>_indices` 字段，返回统计信息。"""
    total_lists = 0
    total_null_removed = 0
    lists_now_empty = 0

    for rec in data["train"]:
        for key, val in rec.items():
            if key.endswith("_indices") and isinstance(val, list):
                total_lists += 1
                before = len(val)
                filtered = [x for x in val if x is not None]
                removed = before - len(filtered)
                total_null_removed += removed
                if removed:
                    rec[key] = filtered
                if len(filtered) == 0 and before > 0:
                    lists_now_empty += 1

    return {
        "total_lists": total_lists,
        "total_null_removed": total_null_removed,
        "lists_now_empty": lists_now_empty,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_PATH, help="要清理的 organ_annotation.json 路径")
    parser.add_argument("--output", default=None, help="输出路径，默认覆盖 --input")
    parser.add_argument("--backup", action="store_true", help="覆盖前先在旁边存一份 <input>.bak")
    parser.add_argument("--dry-run", action="store_true", help="只打印统计信息，不写文件")
    args = parser.parse_args()

    output_path = args.output or args.input

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    stats = clean_null_indices(data)
    print(f"total _indices lists scanned: {stats['total_lists']}")
    print(f"total null entries removed: {stats['total_null_removed']}")
    print(f"lists that became fully empty: {stats['lists_now_empty']}")

    if args.dry_run:
        print("dry-run: 没有写文件")
        return

    if args.backup and output_path == args.input and os.path.exists(args.input):
        backup_path = args.input + ".bak"
        with open(args.input, "r", encoding="utf-8") as f:
            raw = f.read()
        with open(backup_path, "w", encoding="utf-8") as f:
            f.write(raw)
        print(f"backed up original to: {backup_path}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"written: {output_path}")


if __name__ == "__main__":
    main()
