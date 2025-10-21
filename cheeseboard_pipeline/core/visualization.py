import matplotlib.pyplot as plt                         # type: ignore
from matplotlib.patches import Circle,Rectangle         # type: ignore
import os
import numpy as np

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
def plot_trial_paths(start,end,x,y,outfile,title,button=True, arena_center=None, arena_diameter_cm=122.0):
    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    # default arena center if not provided
    if arena_center is None:
        arena_center = (61.0, 61.0)
    
    fig, ax = plt.subplots(figsize=(8, 8)) # legacy step 
    # draw arena (centered at (61,61) with radius 61 cm)
    circ = Circle(arena_center, arena_diameter_cm/2.0, fill=False, lw=3)
    ax.add_patch(circ)
    _add_start_box(ax)

    xs = x[start:end]
    ys = y[start:end]
    ax.plot(xs, ys, lw=1.2, alpha=0.65, color='black')
    # start/end dots (subtle)
    if button:
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

    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)
# ------------------------------------------------
def plot_collapsed_paths_by_blocks(trial_df,session, x, y, outdir, max_blocks=5, use_local=True, session_label="",arena_center=(61,61), arena_diameter_cm=122.0):
    '''
    Make up to 5 figures: one per block (block = contiguous groups of trials, size=block_size).
    Saves files like: block_01.png ... block_05.png
    '''
    os.makedirs(outdir, exist_ok=True)
    
    # choose frame columns
    sf_col = 'start_frame_local' if use_local else 'start_frame'
    ef_col = 'end_frame_local' if use_local else 'stop_frame'

    formats = ['svg','png']  # output formats

    df = trial_df.copy()

    if 'block' not in df.columns:
        df = df.sort_values(sf_col).reset_index(drop=True)
        df = df[df['context'].isin(['wc','bc'])]
        df['block'] = (df['context'] != df['context'].shift()).cumsum()

    blocks = sorted(df['block'].dropna().unique().tolist())
    if max_blocks:
        blocks = blocks[:max_blocks]

    for f in formats:
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
            fig.savefig(os.path.join(outdir, f'{session["Rat"]}_{session["Session"]}_block_{b:02d}_trajectories.{f}'), dpi=300)
            plt.close(fig)
# ------------------------------------------------
def plot_collapsed_paths_by_context(
    trial_df,session,x, y, outdir,
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
    formats = ['svg','png']  # output formats

    # default arena center if not provided
    if arena_center is None:
        arena_center = (61.0, 61.0)

    # only plot trial contexts, skip event rows like '1','2' # legacy step
    df = trial_df.copy()

    for f in formats:
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

                ax.plot(xs, ys, lw=1.2, alpha=0.65, color='black')
                ax.scatter(xs[0],  ys[0],  s=10, color='g', zorder=5)
                ax.scatter(xs[-1], ys[-1], s=10, color='r', marker = 's', zorder=5)
                n_plotted += 1

            # Framing
            r = arena_diameter_cm/2.0
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlim(arena_center[0] - r - margin_cm, arena_center[0] + r + margin_cm)
            ax.set_ylim(arena_center[1] - r - margin_cm, arena_center[1] + r + margin_cm)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('X (cm)')
            ax.set_ylabel('Y (cm)')
            ax.set_title(f'{session_label} Context: {ctx} (n={n_plotted})')

            fig.tight_layout()
            outpath = os.path.join(outdir, f'{session["Rat"]}_{session["Session"]}_context_{ctx}_trajectories.{f}')
            if n_plotted > 0:
                fig.savefig(outpath, dpi=300) # error message
            plt.close(fig)
# ------------------------------------------------
def plot_collapsed_trial_paths(trial_df,session,x,y,outdir,title, use_local=True, arena_center=None, arena_diameter_cm=122.0):
    os.makedirs(outdir, exist_ok=True)
    # default arena center if not provided
    if arena_center is None:
        ac = arena_diameter_cm/2
        arena_center = (ac, ac)
    sf_col = 'start_frame_local' if use_local else 'start frame'
    ef_col = 'end_frame_local' if use_local else 'stop frame'
    formats = ['svg','png']  # output formats

    fig, ax = plt.subplots(figsize=(8, 8))
    # draw arena (centered at (61,61) with radius 61 cm)
    circ = Circle(arena_center, arena_diameter_cm/2.0, fill=False, lw=3)
    ax.add_patch(circ)
    _add_start_box(ax)

    n_plotted = 0
    n_oob = 0   # oob = out of bounds #legacy 
    N = len(x)
    idx = 0
    
    for f in formats:
        outfile = os.path.join(outdir,f'{session["Rat"]}_{session["Session"]}_session_trajectories.{f}')
        for _, row in trial_df.iterrows():
            start = row.get(sf_col,0)
            end = row.get(ef_col,N)
            idx += 1
            try:
                start = int(start)
                end = int(end)
            except Exception as e:
                n_oob += 1
                print(f'Trial {idx} Plotting failed: {e}')
                continue

            if start < 0:
                n_oob += 1
                print(f'Trial {idx} Plotting failed: start < 0')
                continue

            if end > N:
                n_oob += 1
                print(f'Trial {idx} Plotting failed: end > N')
                continue

            if end - start <= 1:
                n_oob += 1
                print(f'Trial {idx} Plotting failed: end - start <= 1')
                continue

            xs = x[start:end]
            ys = y[start:end]

            ax.plot(xs, ys, lw=1.2, alpha=0.65, color='black')
                # start/end dots (subtle)
            if np.isfinite(xs[0]) and np.isfinite(ys[0]):
                ax.scatter(xs[0],  ys[0],  s=10, color='g', zorder=5)
            if np.isfinite(xs[-1]) and np.isfinite(ys[-1]):
                ax.scatter(xs[-1], ys[-1], s=10, color='r', marker = 's', zorder=5)
            n_plotted += 1
        idx = 0 # Important to reset the index for next format 
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-5, arena_diameter_cm + 5)
        ax.set_ylim(-5, arena_diameter_cm + 5)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("X (cm)")
        ax.set_ylabel("Y (cm)")
        ax.set_title(f'{title} plotted={n_plotted}')

        fig.tight_layout()
        fig.savefig(outfile, dpi=300,)
        plt.close(fig)
# ------------------------------------------------
def quick_track_snapshot(x, y, out_png): # debugger snapshot 
    finite = np.isfinite(x) & np.isfinite(y)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot(x[finite], y[finite], lw=0.5, alpha=0.7)
    ax.set_aspect('equal', adjustable='box')
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)