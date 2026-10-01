"""Real chunk-size distribution per max_tokens budget (Docling HybridChunker).

max_tokens is a ceiling, not a target: the HybridChunker chunks by structure
first, then applies a token-aware split + optional merge of undersized peers, so
real chunk sizes form a distribution well below the cap — and a few indivisible
elements (table rows) overshoot it. Sweeping the budget therefore does NOT
linearly sweep real chunk size, so the ablation is only interpretable with the
distribution in hand.

This script tokenises every chunk with the *embedding* tokenizer (BGE-M3, the
one the budget is counted in), reports the size distribution and whether the
over-cap tail is table-driven, and renders an overlaid-density figure.

    uv run python scripts/chunk_size_dist.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer

BUDGETS = [256, 512, 1024]
BASE = Path("data/processed/docling/chunked/hybrid")
CACHE = Path("data/processed/.chunk_dist_cache")
FIG_PATH = Path("docs/adr/assets/chunk_size_distribution.png")
TOKENIZER = "BAAI/bge-m3"
BATCH = 2000

# dataviz categorical slots 1-3 (validated light/dark); text/surface tokens.
COLORS = {256: "#2a78d6", 512: "#008300", 1024: "#e87ba4"}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"


def load(path: Path, tok) -> tuple[np.ndarray, np.ndarray]:
    """Return (token_length, is_table) arrays for every chunk in ``path``."""
    texts, is_table = [], []
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        d = json.loads(line)
        texts.append(d["text"])
        labels = d.get("metadata", {}).get("labels") or []
        is_table.append(any("table" in str(lbl).lower() for lbl in labels))
    lengths: list[int] = []
    for i in range(0, len(texts), BATCH):
        enc = tok(texts[i : i + BATCH], add_special_tokens=False, truncation=False)
        lengths.extend(len(ids) for ids in enc["input_ids"])
    return np.array(lengths), np.array(is_table, dtype=bool)


def load_cached(budget: int, tok) -> tuple[np.ndarray, np.ndarray]:
    """``load`` with a token-length cache keyed by chunk-file mtime (tokenising
    ~600k chunks is the slow part; re-runs for plot tweaks should be instant)."""
    src = BASE / f"chunks_{budget}.jsonl"
    key = CACHE / f"{budget}_{int(src.stat().st_mtime)}.npz"
    if key.exists():
        z = np.load(key)
        return z["lens"], z["is_table"]
    lens, is_table = load(src, tok)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(key, lens=lens, is_table=is_table)
    return lens, is_table


def report(budget: int, lens: np.ndarray, is_table: np.ndarray) -> None:
    over = lens > budget
    lift = (
        100 * is_table[over].mean() / (100 * is_table.mean())
        if is_table.mean() > 0
        else float("nan")
    )
    print(f"\n===== max_tokens = {budget}  ({len(lens):,} chunks) =====")
    print(
        f"  mean={lens.mean():6.1f}  median={np.median(lens):5.0f}  "
        f"p90={np.percentile(lens, 90):4.0f}  p95={np.percentile(lens, 95):4.0f}  max={lens.max():4d}"
    )
    print(
        f"  fill: {100 * np.mean(lens >= 0.9 * budget):5.1f}% >=90% cap | "
        f"{100 * np.mean(lens <= 0.25 * budget):5.1f}% <=25% (tiny) | {100 * over.mean():4.1f}% over cap"
    )
    print(
        f"  tables: {100 * is_table.mean():4.1f}% of all chunks | "
        f"{100 * is_table[over].mean():4.1f}% of OVER-cap chunks  (x{lift:.1f} lift)"
    )


def plot(data: dict[int, np.ndarray]) -> None:
    plt.rcParams.update({"font.size": 11, "axes.edgecolor": INK_MUTED})
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    xmax = 1200
    bins = np.arange(0, xmax + 20, 20)
    for budget in BUDGETS:
        lens = data[budget]
        ax.hist(
            lens,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2.2,
            color=COLORS[budget],
            label=f"max_tokens = {budget}",
        )
        # cap line + direct label in the matching colour (relief for low-contrast hue)
        ax.axvline(budget, color=COLORS[budget], linestyle=":", linewidth=1.6, alpha=0.7)
        ax.text(
            budget,
            ax.get_ylim()[1] * 0.02,
            f" cap {budget}",
            rotation=90,
            va="bottom",
            ha="left",
            color=COLORS[budget],
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xlim(0, xmax)
    ax.set_xlabel("Real chunk size (BGE-M3 tokens)", color=INK_MUTED)
    ax.set_ylabel("Density", color=INK_MUTED)
    ax.set_title(
        "max_tokens is a ceiling, not a target",
        color=INK,
        fontsize=15,
        fontweight="bold",
        loc="left",
        pad=34,
    )
    ax.text(
        0,
        1.055,
        "Real chunk-size distributions overlap heavily; the 1024 budget's mass sits below the 512 cap.",
        transform=ax.transAxes,
        color=INK_MUTED,
        fontsize=10,
    )
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK_MUTED)
    leg = ax.legend(frameon=False, loc="upper right", labelcolor=INK)
    for txt in leg.get_texts():
        txt.set_color(INK)

    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, bbox_inches="tight", facecolor=SURFACE)
    print(f"\nfigure -> {FIG_PATH}")


def main() -> None:
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    data: dict[int, np.ndarray] = {}
    for budget in BUDGETS:
        lens, is_table = load_cached(budget, tok)
        report(budget, lens, is_table)
        data[budget] = lens
    plot(data)


if __name__ == "__main__":
    main()
