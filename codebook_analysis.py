"""
Codebook usage analysis for RPG (static OPQ) and RPGUpgrade_dpqEmbComp (learned DPQ).

Part 1 — Static OPQ (RPG):
    Reads the pre-computed .sem_ids JSON file and plots how frequently each of
    the 256 code entries is used in each of the 32 subspaces.
    Run immediately — no model training required.

Part 2 — Learned DPQ (RPGUpgrade_dpqEmbComp):
    Loads a trained checkpoint, passes all item sentence embeddings through the
    DPQ module at inference mode, and plots the resulting code distribution.
    Requires a trained checkpoint (.pt file).

Usage:
    # Plot OPQ codebook (always available)
    python codebook_analysis.py --method rpg \
        --sem_ids /path/to/text-embedding-3-large_OPQ32,IVF1,PQ32x8_sem_ids.json

    # Plot DPQ codebook (needs a trained checkpoint)
    python codebook_analysis.py --method dpq \
        --checkpoint ckpt/<run_id>/best_model.pt \
        --category Sports_and_Outdoors

    # Plot both side-by-side
    python codebook_analysis.py --method both \
        --sem_ids /path/to/text-embedding-3-large_OPQ32,IVF1,PQ32x8_sem_ids.json \
        --checkpoint ckpt/<run_id>/best_model.pt \
        --category Sports_and_Outdoors
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_code_freq(codes: np.ndarray, n_digits: int = 32, codebook_size: int = 256) -> np.ndarray:
    """
    Args:
        codes: (n_items, n_digits) integer array, values in [0, codebook_size).
    Returns:
        freq: (n_digits, codebook_size) float array of per-digit code counts.
    """
    freq = np.zeros((n_digits, codebook_size), dtype=np.float64)
    for d in range(n_digits):
        freq[d] = np.bincount(codes[:, d], minlength=codebook_size)
    return freq


def compute_stats(freq: np.ndarray) -> dict:
    """Per-digit utilisation rate and entropy (nats)."""
    stats = {}
    n_digits, codebook_size = freq.shape
    for d in range(n_digits):
        row = freq[d]
        utilisation = np.mean(row > 0) * 100.0          # % codes used at least once
        p = row / row.sum()
        p = p[p > 0]
        entropy = -np.sum(p * np.log(p))
        max_entropy = np.log(codebook_size)
        stats[d] = {
            'utilisation': utilisation,
            'entropy_nats': entropy,
            'normalised_entropy': entropy / max_entropy,
        }
    return stats


def print_stats(stats: dict, label: str):
    print(f"\n{'='*60}")
    print(f"  Codebook statistics — {label}")
    print(f"{'='*60}")
    print(f"{'Digit':>6}  {'Util%':>7}  {'Entropy':>9}  {'Norm.H':>7}")
    print(f"{'-'*6}  {'-'*7}  {'-'*9}  {'-'*7}")
    util_vals, norm_h_vals = [], []
    for d, s in stats.items():
        print(f"{d:>6}  {s['utilisation']:>7.1f}  {s['entropy_nats']:>9.4f}  {s['normalised_entropy']:>7.4f}")
        util_vals.append(s['utilisation'])
        norm_h_vals.append(s['normalised_entropy'])
    print(f"\n  Mean utilisation : {np.mean(util_vals):.1f}%")
    print(f"  Mean norm. entropy: {np.mean(norm_h_vals):.4f}  (1.0 = perfectly uniform)")


def plot_heatmap(
    freq: np.ndarray,
    title: str,
    out_path: str,
    n_items: int,
    normalise: bool = True,
    cmap: str = 'viridis',
):
    """
    Heatmap: rows = digit index (0..31), cols = code index (0..255).
    Color encodes relative frequency within each digit row (if normalise=True)
    or raw item count.
    """
    n_digits, codebook_size = freq.shape

    if normalise:
        # Normalise each row independently so colour reflects relative usage
        row_sums = freq.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1          # avoid /0
        display = freq / row_sums
        cbar_label = 'Fraction of items per digit'
    else:
        display = freq
        cbar_label = 'Item count'

    fig_width = max(14, codebook_size // 16)
    fig_height = max(5, n_digits // 4)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    im = ax.imshow(display, aspect='auto', cmap=cmap, interpolation='nearest',
                   vmin=0, vmax=display.max())

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label(cbar_label, fontsize=10)

    ax.set_xlabel('Code index (0 – 255)', fontsize=11)
    ax.set_ylabel('Subspace / digit index', fontsize=11)
    ax.set_title(f'{title}\n({n_items:,} items, {n_digits} subspaces × {codebook_size} codes)', fontsize=12)

    ax.set_yticks(range(n_digits))
    ax.set_yticklabels([str(d) for d in range(n_digits)], fontsize=6)

    # x-axis: tick every 16 codes
    tick_step = 16
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_step))
    ax.tick_params(axis='x', labelsize=7)

    plt.tight_layout()
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_utilisation_bar(
    stats_list: list,          # list of (label, stats_dict)
    out_path: str,
):
    """Bar chart: per-digit utilisation rate for one or two methods."""
    n_digits = len(list(stats_list[0][1].keys()))
    x = np.arange(n_digits)
    width = 0.35 if len(stats_list) == 2 else 0.6
    colors = ['steelblue', 'darkorange']

    fig, ax = plt.subplots(figsize=(14, 4))
    for i, (label, stats) in enumerate(stats_list):
        util = [stats[d]['utilisation'] for d in range(n_digits)]
        offset = (i - (len(stats_list) - 1) / 2) * width
        ax.bar(x + offset, util, width, label=label, color=colors[i], alpha=0.8)

    ax.axhline(100, color='gray', linestyle='--', linewidth=0.8, label='100% (all codes used)')
    ax.set_xlabel('Subspace / digit index')
    ax.set_ylabel('Utilisation (%)')
    ax.set_title('Codebook utilisation per subspace')
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in range(n_digits)], fontsize=7)
    ax.set_ylim(0, 115)
    ax.legend()
    plt.tight_layout()
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_entropy_line(
    stats_list: list,
    out_path: str,
):
    """Line chart: per-digit normalised entropy (0=collapsed, 1=uniform)."""
    n_digits = len(list(stats_list[0][1].keys()))
    x = np.arange(n_digits)
    colors = ['steelblue', 'darkorange']
    markers = ['o', 's']

    fig, ax = plt.subplots(figsize=(14, 4))
    for i, (label, stats) in enumerate(stats_list):
        nh = [stats[d]['normalised_entropy'] for d in range(n_digits)]
        ax.plot(x, nh, marker=markers[i], color=colors[i], label=label, linewidth=1.5, markersize=4)

    ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, label='Maximum (uniform)')
    ax.set_xlabel('Subspace / digit index')
    ax.set_ylabel('Normalised entropy  H / H_max')
    ax.set_title('Codebook diversity per subspace  (1 = perfectly uniform)')
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in range(n_digits)], fontsize=7)
    ax.set_ylim(0, 1.1)
    ax.legend()
    plt.tight_layout()
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# Part 1: Static OPQ codes from .sem_ids file
# ---------------------------------------------------------------------------

def load_sem_ids(path: str) -> np.ndarray:
    """
    Load a .sem_ids JSON (or .json) file.

    Expected format:
        { "<item_asin>": [code0, code1, ..., code31], ... }

    Returns:
        codes: (n_items, n_digits) int32 array
    """
    print(f"  Loading sem_ids from: {path}")
    with open(path, 'r') as f:
        data = json.load(f)
    codes = np.array(list(data.values()), dtype=np.int32)
    print(f"  Loaded {codes.shape[0]:,} items × {codes.shape[1]} digits")
    return codes


def plot_rpg_codebook(sem_ids_path: str, out_dir: str = 'figure'):
    print("\n" + "="*60)
    print("  PART 1 — Static OPQ codebook (RPG)")
    print("="*60)

    codes = load_sem_ids(sem_ids_path)
    n_items, n_digits = codes.shape
    codebook_size = int(codes.max()) + 1
    # Codes may not reach 255; use full 256 for fair comparison
    codebook_size = max(codebook_size, 256)

    freq = compute_code_freq(codes, n_digits=n_digits, codebook_size=codebook_size)
    stats = compute_stats(freq)
    print_stats(stats, label='RPG (static OPQ)')

    plot_heatmap(
        freq, title='RPG — Static OPQ Codebook Usage (fraction)',
        out_path=os.path.join(out_dir, 'RPG_codebook_usage_fraction.png'),
        n_items=n_items,
        normalise=True,
    )
    plot_heatmap(
        freq, title='RPG — Static OPQ Codebook Usage (item count)',
        out_path=os.path.join(out_dir, 'RPG_codebook_usage_count.png'),
        n_items=n_items,
        normalise=False,
    )
    return stats


# ---------------------------------------------------------------------------
# Part 2: Learned DPQ codes from a trained RPGUpgrade_dpqEmbComp checkpoint
# ---------------------------------------------------------------------------

def extract_dpq_codes(checkpoint_path: str, category: str, config_dict: dict = None) -> tuple:
    """
    Load a trained RPGUpgrade_dpqEmbComp checkpoint and run all item sentence
    embeddings through the DPQ module to get discrete code assignments.

    Args:
        checkpoint_path: path to best_model.pt
        category: Amazon Reviews 2014 category (e.g. 'Sports_and_Outdoors')
        config_dict: optional overrides (merged on top of model config)

    Returns:
        codes: (n_items-1, n_digits) int64 numpy array  (item_id 1..n_items-1)
        n_clusters: number of codebook entries (codebook_size) from the loaded model
    """
    import torch
    # Add repo root to sys.path so genrec imports work
    repo_root = os.path.dirname(os.path.abspath(__file__))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from genrec.utils import get_config, get_dataset, get_tokenizer, get_model, init_seed

    overrides = config_dict or {}
    overrides.setdefault('category', category)

    config = get_config(
        model_name='RPGUpgrade_dpqEmbComp',
        dataset_name='AmazonReviews2014',
        config_file=None,
        config_dict=overrides,
    )
    config['device'] = 'cpu'

    # The dataset / tokenizer / model constructors expect config['accelerator']
    # (normally set by Pipeline). Inject a minimal no-op stub here.
    from accelerate import Accelerator
    config['accelerator'] = Accelerator()

    init_seed(config['rand_seed'], config['reproducibility'])

    raw_dataset = get_dataset('AmazonReviews2014')(config)
    tokenizer   = get_tokenizer('RPGUpgrade_dpqEmbComp')(config, raw_dataset)

    model = get_model('RPGUpgrade_dpqEmbComp')(config, raw_dataset, tokenizer)
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    n_clusters = model.dpq.n_clusters

    with torch.no_grad():
        # sent_emb_table.weight: (n_items, d), row 0 = padding
        all_sent = model.sent_emb_table.weight[1:].unsqueeze(0)  # (1, n_items-1, d)
        dpq_out  = model.dpq(all_sent, tau=model.gumbel_tau)      # codes: (1, n_items-1, D)
        codes    = dpq_out['codes'].squeeze(0).cpu().numpy()       # (n_items-1, D)

    print(f"  Extracted DPQ codes: {codes.shape[0]:,} items × {codes.shape[1]} digits")
    print(f"  Code value range: [{codes.min()}, {codes.max()}]")
    return codes, n_clusters


def plot_dpq_codebook(checkpoint_path: str, category: str, out_dir: str = 'figure',
                      config_dict: dict = None):
    print("\n" + "="*60)
    print("  PART 2 — Learned DPQ codebook (RPGUpgrade_dpqEmbComp)")
    print("="*60)

    codes, codebook_size = extract_dpq_codes(checkpoint_path, category, config_dict)
    n_items, n_digits = codes.shape

    freq = compute_code_freq(codes, n_digits=n_digits, codebook_size=codebook_size)
    stats = compute_stats(freq)
    print_stats(stats, label='RPGUpgrade_dpqEmbComp (learned DPQ)')

    plot_heatmap(
        freq, title='RPGUpgrade_dpqEmbComp — Learned DPQ Codebook Usage (fraction)',
        out_path=os.path.join(out_dir, 'RPGUpgrade_dpqEmbComp_codebook_usage_fraction.png'),
        n_items=n_items,
        normalise=True,
    )
    plot_heatmap(
        freq, title='RPGUpgrade_dpqEmbComp — Learned DPQ Codebook Usage (item count)',
        out_path=os.path.join(out_dir, 'RPGUpgrade_dpqEmbComp_codebook_usage_count.png'),
        n_items=n_items,
        normalise=False,
    )
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Visualise codebook usage for RPG (OPQ) and/or RPGUpgrade_dpqEmbComp (DPQ).'
    )
    p.add_argument('--method', choices=['rpg', 'dpq', 'both'], default='rpg',
                   help='Which method to analyse.')
    p.add_argument('--sem_ids',
                   default='/home/trinhhoangkhang/Downloads/'
                           'text-embedding-3-large_OPQ32,IVF1,PQ32x8_sem_ids.json',
                   help='Path to the .sem_ids / _sem_ids.json file (RPG / static OPQ).')
    p.add_argument('--checkpoint', default=None,
                   help='Path to best_model.pt for RPGUpgrade_dpqEmbComp.')
    p.add_argument('--category', default='Sports_and_Outdoors',
                   help='Amazon Reviews 2014 category (used when loading the dataset for DPQ).')
    p.add_argument('--out_dir', default='figure',
                   help='Directory to save output figures.')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    all_stats = []   # list of (label, stats_dict) for comparison plots

    if args.method in ('rpg', 'both'):
        assert args.sem_ids, "--sem_ids is required for method='rpg' or 'both'"
        stats_rpg = plot_rpg_codebook(args.sem_ids, out_dir=args.out_dir)
        all_stats.append(('RPG (static OPQ)', stats_rpg))

    if args.method in ('dpq', 'both'):
        assert args.checkpoint, "--checkpoint is required for method='dpq' or 'both'"
        stats_dpq = plot_dpq_codebook(
            args.checkpoint, args.category, out_dir=args.out_dir
        )
        all_stats.append(('RPGUpgrade_dpqEmbComp (learned DPQ)', stats_dpq))

    # Comparison plots when both methods are available
    if len(all_stats) == 2:
        print("\n  Generating comparison plots…")
        plot_utilisation_bar(
            all_stats,
            out_path=os.path.join(args.out_dir, 'codebook_utilisation_comparison.png'),
        )
        plot_entropy_line(
            all_stats,
            out_path=os.path.join(args.out_dir, 'codebook_entropy_comparison.png'),
        )

    print("\nDone.")


if __name__ == '__main__':
    main()
