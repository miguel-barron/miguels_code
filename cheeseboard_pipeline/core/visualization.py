import matplotlib.pyplot as plt #type: ignore
from matplotlib.patches import Circle,Rectangle
import os
import numpy as np
from scipy.signal import medfilt

def _add_start_box(ax,
                   ll=(0.0, 53.6),  # lower-left (x,y)
                   width=31.4,
                   height=14.8,
                   edge='black',
                   fill='lightgrey',
                   alpha=0.3,
                   z=1):
    '''
    Draws a black-outlined, translucent light grey rectangle on the axes.
    Coordinates are in cm in the same coordinate system as your paths.
    '''
    r = Rectangle(ll, width, height,
                  linewidth=1.5,
                  edgecolor=edge,
                  facecolor=fill,
                  alpha=alpha,
                  zorder=z)
    ax.add_patch(r)
# ------------------------------------------------
def plot_trial_paths(start,end,x,y,outfile,title, arena_center=None, arena_diameter_cm=122.0):
    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    # default arena center if not provided
    if arena_center is None:
        arena_center = (61.0, 61.0)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    # draw arena (centered at (61,61) with radius 61 cm)
    circ = Circle(arena_center, arena_diameter_cm/2.0, fill=False, lw=3)
    ax.add_patch(circ)
    _add_start_box(ax)

#   Cleaning Path
#     xs, ys, diag = _clean_path(x[start:end+1], y[start:end+1],
#                             fps=30.0, cm_per_unit=1.0,          
#                             max_step_cm=8.0, max_speed_cm_s=80.0,
#                             max_turn_deg=170, interp_max_gap=6, smooth_win=15
# )
#     good = np.isfinite(xs) & np.isfinite(ys)
#     for seg in np.split(np.arange(len(xs)), np.where(~good)[0]):
#         if len(seg) > 1 and np.all(good[seg]):
#             ax.plot(xs[seg], ys[seg], lw=1.2, alpha=0.65, color="black")

    xs = x[start:end]
    ys = y[start:end]
    ax.plot(xs, ys, lw=1.2, alpha=0.65, color='black')
        # start/end dots (subtle)
    ax.scatter(xs[0],  ys[0],  s=10, color='g', zorder=5)
    ax.scatter(xs[-1], ys[-1], s=10, color='r', marker = 's', zorder=5)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(arena_center[0] - arena_diameter_cm/2 - 10,
                arena_center[0] + arena_diameter_cm/2 + 10)
    ax.set_ylim(arena_center[1] - arena_diameter_cm/2 - 10,
                arena_center[1] + arena_diameter_cm/2 + 10)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    ax.set_title(title)
   
    # Displaying Clean_Path Diagnosis
    # ax.text(0.02, 0.02,
    #     f"path raw→clean: {diag['raw_path_cm']:.1f}→{diag['clean_path_cm']:.1f} cm\n"
    #     f"spikes: {diag['n_bad']} ({diag['pct_bad']:.1f}%)  "
    #     f"filled: {diag['n_interpolated']}  "
    #     f"remain NaNs: {diag['nans_remaining']}",
    #     transform=ax.transAxes, ha="left", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)
# ------------------------------------------------
def plot_collapsed_paths_by_blocks(trial_df, x, y, outdir, max_blocks=5, use_local=True, session_label="",arena_center=(61,61), arena_diameter_cm=122.0):
    '''
    Make up to 5 figures: one per block (block = contiguous groups of trials, size=block_size).
    Saves files like: block_01.png ... block_05.png
    '''
    os.makedirs(outdir, exist_ok=True)
    # Add a simple block index based on trial order
    # choose frame columns
    sf_col = 'start_frame_local' if use_local else 'start_frame'
    ef_col = 'end_frame_local' if use_local else 'stop_frame'

    df = trial_df.copy()

    if 'block' not in df.columns:
        df = df.sprt_values(sf_col).reset_index(drop=True)
        df = df[df['context'].isin(['wc','bc'])]
        df['block'] = (df[context] != df['context'].shift()).cumsum()

    blocks = sorted(df['block'].dropna().unique().tolist())
    if max_blocks:
        blocks = blocks[:max_blocks]

    for b in blocks:
        block_df = df[df['block'] == b].sort_values(by=sf_col)
        if block_df.empty:
            continue
        context = str(block_df['context'].iloc[0])
        n = len(block_df)

        fig, ax = plt.subplots(figsize=(8, 8))

        # draw arena (centered at (61,61) with radius 61 cm)
        circ = Circle(arena_center, arena_diameter_cm/2.0, fill=False, lw=3)
        ax.add_patch(circ)
        _add_start_box(ax)

        # plot each trial in BLACK
        for _, row in block_df.iterrows():
            start = row.get(sf_col)
            end = row.get(ef_col)
            if np.isnan(start) or np.isnan(end):
                continue
            start = int(start); end = int(end)

            if start < 0 or end <= start:
                continue
            end = min(end, len(x)-1, len(y)-1)
            if end <= start:
                continue
            
            xs = x[start:end+1]
            ys = y[start:end+1]

            # xs, ys, diag = _clean_path(x[start:end+1], y[start:end+1],
            #                 fps=30.0, cm_per_unit=1.0,          
            #                 max_step_cm=8.0, max_speed_cm_s=80,
            #                 max_turn_deg=170, interp_max_gap=6, smooth_win=15
            # )
            # if len(xs) == 0 or len(ys) == 0:
            #     continue

            # good = np.isfinite(xs) & np.isfinite(ys)
            # for seg in np.split(np.arange(len(xs)), np.where(~good)[0]):
            #     if len(seg) > 1 and np.all(good[seg]):
            #         ax.plot(xs[seg], ys[seg], lw=1.2, alpha=0.65, color="black")

            # start/end dots (subtle)
            ax.plot(xs, ys, lw=1.2, alpha=0.65, color='black')
            ax.scatter(xs[0],  ys[0],  s=10, color='g', zorder=5)
            ax.scatter(xs[-1], ys[-1], s=10, color='r', marker = 's', zorder=5)

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(arena_center[0] - arena_diameter_cm/2 - 10,
                    arena_center[0] + arena_diameter_cm/2 + 10)
        ax.set_ylim(arena_center[1] - arena_diameter_cm/2 - 10,
                    arena_center[1] + arena_diameter_cm/2 + 10)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Y (cm)')
        ax.set_title(f'{session_label} Block {b} ({context}) (n={n})')

        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f'block_{b:02d}.png'), dpi=150)
        plt.close(fig)
# ------------------------------------------------
def plot_collapsed_paths_by_context(
    trial_df, x, y, outdir,
    contexts=('wc', 'bc'), use_local=True, session_label='',
    arena_center=None, arena_diameter_cm=122.0, margin_cm=10.0
):
    '''
    Make up to 2 figures: one for wc and one for bc.
    Saves files like: context_wc.png / context_bc.png
    '''

    os.makedirs(outdir, exist_ok=True)

    # pick frame columns by name
    sf_col = 'start_frame_local' if use_local else 'start_frame'
    ef_col = 'end_frame_local' if use_local else 'stop_frame'

    # default arena center if not provided
    if arena_center is None:
        arena_center = (61.0, 61.0)

    # only plot trial contexts, skip event rows like '1','2'
    df = trial_df.copy()

    for ctx in contexts:
        ctx_df = df[df['context'] == ctx].sort_values(by=sf_col)
        if ctx_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 8))

        # arena outline
        circ = Circle(arena_center, arena_diameter_cm/2.0, fill=False, lw=3)
        ax.add_patch(circ)
        _add_start_box(ax)

        n_plotted = 0
        for _, row in ctx_df.iterrows():
            start = row.get(sf_col)
            end   = row.get(ef_col)

            # guard NaNs / bad ranges
            if start is None or end is None:
                continue
            if np.isnan(start) or np.isnan(end):
                continue

            start = int(start); end = int(end)
            if start < 0 or end <= start:
                continue

            # clamp to array bounds
            end = min(end, len(x) - 1, len(y) - 1)
            if end <= start:
                continue

            xs = x[start:end+1]
            ys = y[start:end+1]

            # # optional de-spike / smoothing if you have it
            # try:
            #     xs, ys, diag = _clean_path(x[start:end+1], y[start:end+1],
            #                 fps=30.0, cm_per_unit=1.0,          
            #                 max_step_cm=8.0, max_speed_cm_s=80.0,
            #                 max_turn_deg=170, interp_max_gap=6, smooth_win=15
            # )
            # except NameError:
            #     pass  # no _clean_path available; just plot raw slices

            # if len(xs) == 0 or len(ys) == 0:
            #     continue

            # good = np.isfinite(xs) & np.isfinite(ys)
            # for seg in np.split(np.arange(len(xs)), np.where(~good)[0]):
            #     if len(seg) > 1 and np.all(good[seg]):
            #         ax.plot(xs[seg], ys[seg], lw=1.2, alpha=0.65, color='black')
            ax.plot(xs, ys, lw=1.2, alpha=0.65, color='black')
            ax.scatter(xs[0],  ys[0],  s=10, color='g', zorder=5)
            ax.scatter(xs[-1], ys[-1], s=10, color='r', marker = 's', zorder=5)
            n_plotted += 1

        # framing
        r = arena_diameter_cm/2.0
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(arena_center[0] - r - margin_cm, arena_center[0] + r + margin_cm)
        ax.set_ylim(arena_center[1] - r - margin_cm, arena_center[1] + r + margin_cm)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Y (cm)')
        ax.set_title(f'{session_label} Context: {ctx} (n={n_plotted})')

        fig.tight_layout()
        outpath = os.path.join(outdir, f'context_{ctx}.png')
        if n_plotted > 0:
            fig.savefig(outpath, dpi=300)
        plt.close(fig)
# ------------------------------------------------
def _interp_nans_limited(a, max_gap=6):
    '''
    Linearly interpolate interior NaNs only if the run length ≤ max_gap samples.
    Longer runs are left as NaN so they don't fabricate long straight lines.
    Leading/trailing NaNs are preserved.
    '''
    a = np.asarray(a, float)
    n = len(a)
    if n == 0:
        return a
    good = np.isfinite(a)
    if good.sum() == 0:
        return a.copy()

    out = a.copy()
    i = 0
    while i < n:
        if np.isfinite(out[i]):
            i += 1
            continue
        j = i
        while j < n and not np.isfinite(out[j]):
            j += 1
        # now [i, j-1] is a NaN run
        if i > 0 and j < n and (j - i) <= max_gap:
            out[i:j] = np.interp(np.arange(i, j), [i-1, j], [out[i-1], out[j]])
        # else: leave NaNs
        i = j
    return out
# ------------------------------------------------
def _moving_average(a, w=5, median_first=True):
    '''
    Light smoothing; median filter first to suppress spikes, then MA.
    '''
    a = np.asarray(a, float)
    if median_first and w >= 3 and (w % 2 == 1):
        a = medfilt(a, kernel_size=w)
    if w <= 1:
        return a
    k = np.ones(w) / w
    pad = w // 2
    ap = np.pad(a, (pad, pad), mode='edge')
    return np.convolve(ap, k, mode='valid')
# ------------------------------------------------
def _despike_xy(x, y, fps, cm_per_unit=1.0,
                max_step_cm=8.0, max_speed_cm_s=120.0, max_turn_deg=170):
    '''
    Mark samples as NaN if:
      - step length > max_step_cm (instantaneous jump), OR
      - speed > max_speed_cm_s, OR
      - direction reverses unrealistically (turn angle > max_turn_deg) while step is large.
    Returns (x2, y2, diag) with diag counters.
    '''
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = len(x)
    dx = np.diff(x); dy = np.diff(y)
    step = np.hypot(dx, dy) * cm_per_unit
    speed = step * fps

    # turn angle (in degrees) using three consecutive points
    ux = np.r_[np.nan, dx]; uy = np.r_[np.nan, dy]
    v1 = np.c_[ux[:-1], uy[:-1]]
    v2 = np.c_[ux[1:],  uy[1:]]
    dot = np.einsum('ij,ij->i', v1, v2)
    n1 = np.hypot(v1[:,0], v1[:,1]); n2 = np.hypot(v2[:,0], v2[:,1])
    cosang = np.clip(dot / (n1 * n2 + 1e-12), -1, 1)
    ang = np.degrees(np.arccos(cosang))          # angle between successive steps
    # align to points k (second point of each step)
    ang = np.r_[np.nan, ang]

    # bad if any criterion trips
    bad_step = step > max_step_cm
    bad_speed = speed > max_speed_cm_s
    # large turn *and* a decent step (ignore tiny jitter)
    bad_turn = (ang > max_turn_deg) & (np.r_[False, step > (0.5 * max_step_cm)])

    bad = bad_step | bad_speed
    bad = np.r_[False, bad] | bad_turn

    x2 = x.copy(); y2 = y.copy()
    n_bad = int(np.nansum(bad))
    x2[bad] = np.nan; y2[bad] = np.nan
    diag = {
        'n': n,
        'n_bad': n_bad,
        'pct_bad': 100.0 * n_bad / max(n, 1),
        'n_bad_step': int(np.nansum(np.r_[False, bad_step])),
        'n_bad_speed': int(np.nansum(np.r_[False, bad_speed])),
        'n_bad_turn': int(np.nansum(bad_turn)),
        'max_step_cm': float(np.nanmax(step)) if len(step) else np.nan,
        'max_speed_cm_s': float(np.nanmax(speed)) if len(speed) else np.nan,
    }
    return x2, y2, diag
# ------------------------------------------------
def _clean_path(x, y, fps=30.0, cm_per_unit=1.0,
                max_step_cm=8.0, max_speed_cm_s=120.0,
                max_turn_deg=170, interp_max_gap=6, smooth_win=5):
    '''
    Pipeline:
      1) speed/step/turn-based despike → NaNs (keep evidence of long dropouts)
      2) gap-limited interpolation (≤ interp_max_gap samples)
      3) light median+moving-average smoothing
    Returns xs, ys, diagnostics dict (raw vs cleaned path, % interpolated, etc.)
    '''
    x = np.asarray(x, float); y = np.asarray(y, float)
    # raw stats
    raw_step = np.hypot(np.diff(x), np.diff(y)) * cm_per_unit
    raw_path = float(np.nansum(raw_step))

    x1, y1, d1 = _despike_xy(x, y, fps, cm_per_unit,
                              max_step_cm=max_step_cm,
                              max_speed_cm_s=max_speed_cm_s,
                              max_turn_deg=max_turn_deg)

    x2 = _interp_nans_limited(x1, max_gap=interp_max_gap)
    y2 = _interp_nans_limited(y1, max_gap=interp_max_gap)

    xs = _moving_average(x2, w=smooth_win)
    ys = _moving_average(y2, w=smooth_win)

    # cleaned stats
    clean_step = np.hypot(np.diff(xs), np.diff(ys)) * cm_per_unit
    clean_path = float(np.nansum(clean_step))
    n_interped = int(np.sum(~np.isfinite(x1)) - np.sum(~np.isfinite(x2)))  # filled points
    total_nans = int(np.sum(~np.isfinite(x1)))

    diag = {
        **d1,
        'raw_path_cm': raw_path,
        'clean_path_cm': clean_path,
        'path_reduction_cm': raw_path - clean_path,
        'n_interpolated': n_interped,
        'nans_remaining': int(np.sum(~np.isfinite(x2) | ~np.isfinite(y2))),
        'nans_total_after_despike': total_nans,
    }
    return xs, ys, diag