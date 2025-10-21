
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------- Configuration you can tweak ----------
DAYS = [1, 10, 14]
METRICS = [
    ('total path', 'Total Path Length (cm)', 'total_path'),
    ('path to well', 'Path to Well (cm)', 'path_to_well'),
    ('trial time', 'Trial Time (s)', 'trial_time'),
]

# If your contexts use different labels, map them here → pretty labels
CONTEXT_LABELS = {
    'wc': 'White Curtain',
    'bc': 'Black Curtain',
    'white': 'White Curtain',
    'black': 'Black Curtain',
}
# ------------------------------------------------
def make_metric_figure(df, metric, ylabel, outdir, fname_stub):
    '''
    One figure with three panels:
      - Day 1 (x = Block 1..5)
      - Day 10 (x = Block 1..5)
      - Day 14 (x = Context)
    '''
    # no horizontal gaps between panels
    fig, axes = plt.subplots(
        1, 3, figsize=(13, 5.2), sharey=True,
        gridspec_kw={'wspace': 0.0}
    )
    fig.subplots_adjust(wspace=0.0, left=0.08, right=0.98, bottom=0.12, top=0.9)

    SLOT_POS = [1, 2, 3, 4, 5]
    CONTEXT_SLOTS = {'Black Curtain': 2, 'White Curtain': 4}
    FULL_BLOCKS = [f'Block {i}' for i in range(1, 6)]
    subplot_idx = 1
    for i_ax, (ax, day) in enumerate(zip(axes, DAYS)):
        sub = df[df['day'] == day]
        ax.set_xlim(0.5, 5.5)                     # identical width across panels
        ax.set_xlabel(f'Day {day}')
        show_y = (i_ax == 0)
        
        # categories: blocks for day 1/10; contexts for day 14
        if day in (1, 10):
            # 5 block slots; may be empty
            x_cats = FULL_BLOCKS
            positions = SLOT_POS
            # ticks: label all 5 blocks
            tick_pos = SLOT_POS
            tick_lab = FULL_BLOCKS
        else:
            # Day 14: map contexts to fixed slots (2 and 4 here)
            present_ctx = (sub['group_var'].dropna().unique().tolist())
            # keep order Black, White if present
            ordered_ctx = [c for c in ['Black Curtain', 'White Curtain'] if c in present_ctx]
            x_cats = ordered_ctx
            positions = [CONTEXT_SLOTS[c] for c in ordered_ctx]
            # ticks: show labels only at used slots; blanks elsewhere
            tick_pos = SLOT_POS
            tick_lab = ['', '', '', '', '']
            for c, p in zip(ordered_ctx, positions):
                tick_lab[p-1] = c  # p is 1-based

        draw_box_with_jitter(
            ax=ax,
            x_categories=x_cats,
            x_positions=positions,
            tick_positions=tick_pos,
            tick_labels=tick_lab,
            data_series=sub[metric],
            cat_series=sub['group_var'],
            ylabel=ylabel if show_y else '',
            xlabel=f'Day {day}',
            idx = subplot_idx,
            context_series=sub.get('context', None),
        )
        subplot_idx += 1

    outpath = Path(outdir) / f'{fname_stub}.svg'
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'[Saved] {outpath}')
# ------------------------------------------------
def draw_box_with_jitter(ax, x_categories, x_positions, tick_positions, tick_labels, data_series, 
                         cat_series, ylabel,xlabel,idx, context_series=None,):
    '''
    Draw a boxplot per category in x_categories, plus jittered points (to the RIGHT of the box).
    - Boxes are filled by dominant context in that category: wc=white, bc=black.
    - Mean is drawn as a circle: wc=black circle, bc=white circle.
    '''
    import numpy as np
    wc = 40.92 #cm
    bc = 198.12 #cm
    # helper to pick a dominant context for a category
    def dominant_context_for_cat(cat):
        if context_series is None:
            return None
        mask = (cat_series == cat)
        if mask.sum() == 0:
            return None
        # normalize to lower for mapping
        vals = context_series[mask].astype(str).str.lower()
        # choose wc/bc if present; otherwise None
        counts = vals.value_counts()
        if 'wc' in counts.index or 'white' in counts.index:
            wc_count = counts.get('wc', 0) + counts.get('white', 0)
        else:
            wc_count = 0
        if 'bc' in counts.index or 'black' in counts.index:
            bc_count = counts.get('bc', 0) + counts.get('black', 0)
        else:
            bc_count = 0
        if wc_count == bc_count == 0:
            # fallback to most frequent label
            return counts.index[0] if len(counts) else None
        return 'wc' if wc_count >= bc_count else 'bc'

    # draw each category separately so we can color individually & skip empties
    for pos, cat in zip(x_positions, x_categories):
        ys = data_series[cat_series == cat].dropna().values
        if ys.size == 0:
            continue  # keep tick but no box

        # boxplot for this single category
        bp = ax.boxplot(
            [ys],
            positions=[pos],
            widths=0.18,
            showmeans=False,
            meanline=False,
            patch_artist=True,
            manage_ticks=False,
            showcaps=False,
            sym='' 
        )

        # choose colors by dominant context
        dom = dominant_context_for_cat(cat)
        metric_name = (getattr(data_series, "name", "") or "").strip().lower()

        # wc = white box; bc = black box; otherwise light grey
        if dom in ('wc', 'white'):
            box_face = 'white'
            mean_face, mean_edge = 'black', 'black'  # black circle
            if metric_name == "path to well":
                ideal_y = wc / 2
            elif metric_name == "total path":
                ideal_y = wc
            else:
                ideal_y = None
        elif dom in ('bc', 'black'):
            box_face = 'black'
            mean_face, mean_edge = 'white', 'white'  # white circle
            if metric_name == "path to well":
                ideal_y = bc / 2
            elif metric_name == "total path":
                ideal_y = bc
            else:
                ideal_y = None
        else:
            box_face = "#DDDDDD"
            mean_face, mean_edge = 'black', 'black'
            ideal_y = None

        for patch in bp['boxes']:
            patch.set_facecolor(box_face)
            patch.set_edgecolor('black')
        for whisk in bp['whiskers']:
            whisk.set_color('black')
        for med in bp['medians']:
            med.set_visible(False)

        # scatter points: to the RIGHT of the box with small jitter
        if ys.size:
            # right offset ~0.22; tiny jitter [0, 0.06]
            xs = pos + 0.22 + np.random.rand(len(ys)) * 0.06
            ax.plot(xs, ys, linestyle='none', marker='o',
                    alpha=0.9, markersize=3.5, c='k')

        # draw the mean as a circle on the box center
        mean_y = float(np.mean(ys))
        ax.plot([pos], [mean_y], marker='o', markersize=3.5,
                markerfacecolor=mean_face, markeredgecolor=mean_edge, zorder=3)
        if ideal_y is not None:
            ax.plot([pos], [ideal_y], marker='*', markersize=3.5,
                markerfacecolor='r', markeredgecolor='r', zorder=3)

    # axis cosmetics
    ax.set_xlabel(xlabel)
    ax.xaxis.labelpad = 10
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if idx > 1:
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)
    else:
        ax.set_ylabel(ylabel)
# ------------------------------------------------
def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'day' not in df.columns:
        raise ValueError('Input data must include a "day" column.')
    if 'block' not in df.columns:
        raise ValueError('Input data must include a "block" column for days 1 and 10.')
    if 'context' not in df.columns:
        raise ValueError('Input data must include a "context" column for day 14.')

    # Keep only successful trials and the requested days
    df = df[df['day'].isin(DAYS)].copy()

    # Standardize context labels (only matters for day 14)
    df['context_std'] = (
        df['context']
        .astype(str)
        .str.strip()
        .str.lower()
        .map(lambda x: CONTEXT_LABELS.get(x, x.title()))
    )

    # Build a unified group variable per day:
    # - Day 1 & 10: group by Block N
    # - Day 14: group by Context label
    def _group_label(row):
        d = int(row['day'])
        if d in (1, 10):
            # coerce block to int if possible, else string
            try:
                b = int(row['block'])
            except Exception:
                b = str(row['block'])
            return f'Block {b}'
        else:
            return str(row['context_std'])

    df['group_var'] = df.apply(_group_label, axis=1)
    df['day_label'] = df['day'].map(lambda d: f'Day {int(d)}')
    return df
# ------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Make three figures (one per metric) with Day 1/10 by Block and Day 14 by Context."
    )
    parser.add_argument("csv", help="Path to input CSV with trial-level data.")
    parser.add_argument(
        "-o", "--outdir", default="figures_boxplots",
        help="Directory to save figures (default: figures_boxplots)"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df = prepare_df(df)

    # Ensure required metric columns exist
    missing = [m for m, _, _ in METRICS if m not in df.columns]
    if missing:
        raise ValueError(f"Missing required metric columns: {missing}")

    for metric, ylabel, stub in METRICS:
        make_metric_figure(df, metric, ylabel, args.outdir, f"{stub}_by_day_panels")


if __name__ == "__main__":
    main()