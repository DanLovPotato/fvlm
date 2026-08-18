"""Visualize per-organ-pathology AUC trends across epochs from eval_finetune.py runs.

Reads eval_finetune.py's stdout logs (one file per epoch, expected to contain the
"  {organ}_{pathology}: n=... base_rate=... accuracy=... auc=..." lines printed by
score_against_ground_truth()) and plots AUC vs epoch, one subplot per organ, one line
per pathology item. Frozen organs (FROZEN_ORGANS, never trained by finetune.py's
adapter) are marked in each subplot title so a flat line there reads as "expected",
not "broken".

Usage:
    Edit LOG_DIR / OUTPUT_PATH below, then: python plot_eval_finetune_trends.py

Log files are matched by filename pattern epoch_<NNN>.log (NNN used as the x-axis
value, in whatever units the caller named the checkpoints - e.g. epoch_010.log ->
x=10) and glob-sorted numerically. epoch_000.log is expected to be the untrained
baseline (eval_finetune.py --untrained: expand_organs() applied with no finetuned
checkpoint loaded) rather than an actual trained checkpoint - checkpoint_000.pth is
already one epoch further along - so x=0 is marked distinctly (star marker, dashed
vertical reference line, "untrained" x-tick label) rather than plotted as if it were
just another training epoch.
"""
import glob
import math
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Lives in _graph/ (kept separate from the top-level pipeline scripts), but needs
# finetune.py's ORGANS/FROZEN_ORGANS from the repo root - add it to sys.path so this
# runs regardless of cwd, without requiring callers to set PYTHONPATH themselves.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from finetune import FROZEN_ORGANS, ORGANS

# Directory containing eval_finetune.py's epoch_<N>.log stdout captures.
LOG_DIR = "/mnt/researchdrive/ptiwari9/Staff_Trainee_Folders/Dan/chestCT/weights/fvlm_weights/eval_finetune_logs"
# Where to save the resulting PNG.
OUTPUT_PATH = "/home/wisc/dxiang23/Projects/chestCT/code/fvlm/_graph/eval_finetune_trends.png"

LINE_RE = re.compile(
    r"^\s*(?P<item>.+?): n=(?P<n>\d+) base_rate=(?P<base_rate>[\d.]+) "
    r"accuracy=(?P<acc>[\d.]+) auc=(?P<auc>[\d.]+|nan)\s*$"
)
LOG_NAME_RE = re.compile(r"epoch_(\d+)\.log$")


def parse_logs(log_dir):
    """-> {item: {epoch_int: auc_float_or_nan}}, epochs (sorted list of int)."""
    data = {}
    epochs = []
    for path in sorted(glob.glob(os.path.join(log_dir, "epoch_*.log"))):
        m = LOG_NAME_RE.search(os.path.basename(path))
        if not m:
            continue
        epoch = int(m.group(1))
        epochs.append(epoch)
        with open(path) as f:
            for line in f:
                m = LINE_RE.match(line)
                if not m:
                    continue
                item = m.group("item")
                auc = float(m.group("auc")) if m.group("auc") != "nan" else float("nan")
                data.setdefault(item, {})[epoch] = auc
    epochs = sorted(set(epochs))
    return data, epochs


def organ_of(item):
    # item is "{organ}_{pathology}" - organ names never contain "_" (they use
    # spaces, e.g. "trachea and bronchie"), so splitting on the first "_" is safe.
    return item.split("_", 1)[0]


def plot_trends(data, epochs, output_path):
    organs = [o for o in ORGANS if any(organ_of(item) == o for item in data)]
    n_organs = len(organs)
    n_cols = 3
    n_rows = math.ceil(n_organs / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)

    # Epochs are irregularly spaced (0, 1, 5, 10, 15, ...) - plotting at their literal
    # numeric values crams 0 and 1 together at the left edge with a big empty gap
    # after. Plot at evenly-spaced positions instead and label the ticks with the real
    # epoch numbers, so every point (including the "untrained" one) gets equal room.
    x_positions = list(range(len(epochs)))

    for idx, organ in enumerate(organs):
        ax = axes[idx // n_cols][idx % n_cols]
        is_frozen = organ in FROZEN_ORGANS
        items = sorted(item for item in data if organ_of(item) == organ)

        for item in items:
            pathology = item.split("_", 1)[1]
            ys = [data[item].get(e, float("nan")) for e in epochs]
            line, = ax.plot(x_positions, ys, marker="o", markersize=3, label=pathology)
            # x=0 is the untrained baseline, not a trained checkpoint - mark it with a
            # bigger star in the same color so it reads as "starting point", not just
            # another data point on the training curve.
            if 0 in epochs and not math.isnan(data[item].get(0, float("nan"))):
                ax.plot(0, data[item][0], marker="*", markersize=12,
                         color=line.get_color(), markeredgecolor="black", markeredgewidth=0.5)

        if 0 in epochs:
            ax.axvline(0, color="black", linestyle=":", linewidth=1, alpha=0.5)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance (0.5)")
        ax.set_title(f"{organ}{' [FROZEN]' if is_frozen else ' [trainable]'}")
        ax.set_xlabel("epoch")
        ax.set_ylabel("AUC")
        ax.set_ylim(0.2, 0.9)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(["untrained" if e == 0 else str(e) for e in epochs], rotation=45, ha="right")
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
        ax.grid(alpha=0.3)

    for idx in range(n_organs, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    fig.suptitle("eval_finetune.py: zero-shot pathology AUC vs. training epoch", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved {output_path}")


def main():
    data, epochs = parse_logs(LOG_DIR)
    if not data:
        raise SystemExit(f"No epoch_*.log files with parseable AUC lines found under {LOG_DIR}")
    print(f"Parsed {len(data)} items across {len(epochs)} epochs: {epochs}")
    plot_trends(data, epochs, OUTPUT_PATH)


if __name__ == "__main__":
    main()
