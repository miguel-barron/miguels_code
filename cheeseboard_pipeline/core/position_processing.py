import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import medfilt
import core.trial_extraction as extract
import matplotlib.pyplot as plt

def normalize_frames(dlc_df, frame_col="frame_idx"):
    """Return DLC with a continuous, sorted frame index and float tracking cols."""
    if frame_col not in dlc_df.columns:
        raise KeyError(f"Missing '{frame_col}' in DLC dataframe")
    df = dlc_df.copy()

    # Ensure integer-ish frame column and sort
    df[frame_col] = pd.to_numeric(df[frame_col], errors="coerce")
    df = df.dropna(subset=[frame_col]).sort_values(frame_col)
    df[frame_col] = df[frame_col].astype(np.int64)

    fmin, fmax = int(0), int(df[frame_col].max())
    full_idx = pd.Index(range(fmin, fmax + 1), name=frame_col)

    # Reindex once so ALL columns share the same frame basis
    df = df.set_index(frame_col).reindex(full_idx)

    # Cast tracking/likelihood to float (so NaN is allowed)
    float_cols = [c for c in df.columns if c.endswith((".x", ".y", ".score"))]
    df[float_cols] = df[float_cols].astype(float)

    return df.reset_index()
# ------------------------------------------------
def dlc_arrays(df, cols=("lear.x","lear.y","rear.x","rear.y","lear.score","rear.score")):
    """
    Extracts numpy arrays for the requested cols and asserts equal length.
    """
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"DLC missing columns: {missing}")

    arrs = {c: df[c].to_numpy() for c in cols}
    lens = {len(v) for v in arrs.values()}
    if len(lens) != 1:
        detail = {k: len(v) for k, v in arrs.items()}
        raise ValueError(f"DLC columns have unequal lengths: {detail}")
    return arrs
# ------------------------------------------------
def slice_dlc_to_trials(
    dlc_df,
    trial_df,
    frame_col="frame_idx",
    start_col="start frame",
    stop_col="stop frame"
):
    """
    Normalize DLC to continuous frames, then keep only rows inside any trial interval.
    Avoids column-position assumptions; handles missing frames.
    """
    # 1) Normalize DLC frames
    df = normalize_frames(dlc_df, frame_col=frame_col)

    # 2) Validate trial columns and coerce to integers
    if start_col not in trial_df.columns or stop_col not in trial_df.columns:
        raise KeyError(f"Trial DF missing '{start_col}' and/or '{stop_col}'")

    starts = pd.to_numeric(trial_df[start_col], errors="coerce").dropna().astype(np.int64).to_numpy()
    stops  = pd.to_numeric(trial_df[stop_col],  errors="coerce").dropna().astype(np.int64).to_numpy()
    if len(starts) != len(stops):
        raise ValueError("Starts/stops length mismatch after coercion")

    # 3) Build a boolean mask over the full frame range
    frames = df[frame_col].to_numpy()  # continuous, sorted
    fmin, fmax = int(frames.min()), int(frames.max())
    n = fmax - fmin + 1
    keep = np.zeros(n, dtype=bool)

    # Clip intervals to available range and mark them True
    for s, e in zip(starts, stops):
        if not np.isfinite(s) or not np.isfinite(e):
            continue
        s = int(s); e = int(e)
        if e < fmin or s > fmax:    # completely outside
            continue
        s = max(s, fmin)
        e = min(e, fmax)
        if e >= s:
            keep[(s - fmin):(e - fmin + 1)] = True

    # 4) Apply mask
    df_trimmed = df[keep].reset_index(drop=True)
    return df_trimmed
# ------------------------------------------------
def scale_to_arena(
    x, y,
    arena_size,
    center,
    flip_y=True,
    extent_quantiles=(0.01, 0.99),
    min_extent_px=1e-3,
    well_px=None,           # (x, y) of second well in pixels
    well_dist_cm=None       # known center→well distance in cm
):
    '''
    Scale raw coordinates to arena coordinates in cm.

    Parameters
    ----------
    x, y : np.ndarray
        Raw coordinate arrays.
    arena_size : float
        Arena diameter in cm.
    center : tuple(float, float)
        Pixel coordinates (center_x, center_y) of the arena center.
    flip_y : bool
        Whether to flip the y-axis to match physical orientation.
    extent_quantiles : tuple(float, float)
        Quantiles used to determine pixel extents if no well reference is provided.
    min_extent_px : float
        Minimum allowed pixel extent to avoid divide-by-zero.
    well_px : tuple(float, float), optional
        Pixel coordinates of a known reference well (e.g., outer ring).
    well_dist_cm : float, optional
        Known real-world distance (cm) between arena center and that well.
        If provided along with well_px, the function uses this to determine scale.
    '''
    rad = arena_size / 2.0

    x = x.astype(float)
    y = y.astype(float)

    # plt.hist(y,bins=10)
    # plt.show()

    # Validate/repair center
    cx, cy = center if center is not None else (np.nan, np.nan)
    if not np.isfinite(cx) or not np.isfinite(cy) or abs(cx) > 1e6 or abs(cy) > 1e6:
        fx = np.isfinite(x); fy = np.isfinite(y)
        if (fx & fy).any():
            cx = np.nanmedian(x[fx])
            cy = np.nanmedian(y[fy])
        else:
            cx = 0.0; cy = 0.0

    # Center and optional flip
    x_c = x - cx
    y_c = y - cy
    if flip_y:
        y_c = -y_c

    # Finite mask # Verify again 
    f = np.isfinite(x_c) & np.isfinite(y_c)
    if not f.any():
        return np.full_like(x_c, np.nan), np.full_like(y_c, np.nan)

    # --- Option A: Use well reference for scaling ---
    if well_px is not None and well_dist_cm is not None:
        wx, wy = well_px
        # Distance from arena center to well in pixels
        pix_dist = np.hypot(wx - cx, wy - cy)
        if pix_dist < min_extent_px: # TO CHECK 
            pix_dist = min_extent_px
        scale = well_dist_cm / pix_dist
        sx = sy = scale
    else:
        # --- Option B: Default quantile-based extent scaling ---
        x_f = x_c[f]; y_f = y_c[f]
        qlo, qhi = extent_quantiles
        try:
            x_lo, x_hi = np.quantile(x_f, [qlo, qhi]); ext_x = max(x_hi - x_lo, 0.0)
            # x_lo 1% and x_hi 99% of x-values lie
            # effective pixel width of the arena
        except Exception:
            ext_x = np.nan
        try:
            y_lo, y_hi = np.quantile(y_f, [qlo, qhi]); ext_y = max(y_hi - y_lo, 0.0)
            # y_lo 1% and y_hi 99% of y-values lie
            # effective pixel height of the arena
        except Exception:
            ext_y = np.nan

        if not np.isfinite(ext_x) or ext_x < min_extent_px:
            ext_x = np.nanmax(x_f) - np.nanmin(x_f)
            if not np.isfinite(ext_x) or ext_x < min_extent_px:
                ext_x = 1.0
        if not np.isfinite(ext_y) or ext_y < min_extent_px:
            ext_y = np.nanmax(y_f) - np.nanmin(y_f)
            if not np.isfinite(ext_y) or ext_y < min_extent_px:
                ext_y = 1.0

        sx = arena_size / ext_x
        sy = arena_size / ext_y

    # Apply scaling and arena recenteri
    x_scaled = x_c * sx + rad
    y_scaled = y_c * sy + rad
    return x_scaled, y_scaled
# ------------------------------------------------
def remove_outliers(arr, z_thresh=4):
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    mask = np.abs(arr - mean) < z_thresh * std
    arr[~mask] = np.nan
    return arr
# ------------------------------------------------
def smooth_positions(x, y, window=5):
    return uniform_filter1d(x, size=window), uniform_filter1d(y, size=window)
# ------------------------------------------------
def calculate_velocity(x, y):
    vel = np.sqrt(np.diff(x, prepend=x[0])**2 + np.diff(y, prepend=y[0])**2)
    return vel
# ------------------------------------------------
def calculate_head_direction(pos):
    dx = pos[:, 0] - pos[:, 2]
    dy = pos[:, 1] - pos[:, 3]
    hd = np.arctan2(dy, dx) * 180 / np.pi
    return (hd + 360) % 360
# ------------------------------------------------
def compute_trial_metrics(trial_df, experiment, tracking_data,cap_sec,fps=30):
    x, y, hd = tracking_data['x'], tracking_data['y'], tracking_data['head_direction']
    N = len(x)
    CAP = int(fps * cap_sec)
    metrics_list = []
    metric_indices = []
    def safe_index(value):
        if value is None or pd.isna(value):
            return None
        return int(value)

    if experiment == 'CH_3day':
        for idx,row in trial_df.iterrows():
            start = row.get('start_frame_local', None)
            stop = row.get('stop_frame_local', None)
            
            well_1 = safe_index(row.get('well_1_frame_local', None))
            well_2 = safe_index(row.get('well_2_frame_local', None))
            well_3 = safe_index(row.get('well_3_frame_local', None))

            if start is None or stop is None or pd.isna(start) or pd.isna(stop):
                print(f'[WARN] Trial {idx}: Missing local index — skipping.')
                continue

            start = int(start)
            stop = int(stop)

            end_cap = start + CAP
            end_stop = stop + 1
            end = min(end_cap, end_stop)

            if start < 0:
                start = 0
            if end > N:
                end = N

            if end - start <= 1:
                print(f'[WARN] Trial {idx}: end - start <= 1 -- Skipping Trial')

            path_total = extract.compute_path_length(x, y, start, stop)
            path_to_well_1 = (extract.compute_path_length(x, y, start, well_1)
                                if well_1 is not None else None)
            path_to_well_2 = (extract.compute_path_length(x, y, start, well_2)
                                if well_2 is not None else None)
            path_to_well_3 = (extract.compute_path_length(x, y, start, well_3)
                                if well_3 is not None else None)
            head_dir,hd_frame = extract.get_trial_head_direction(hd, start)

            metrics_list.append({
                'end_frame_local':end,
                'head direction': head_dir,
                'hd frame': hd_frame,
                'path to well 1': path_to_well_1,
                'path to well 2': path_to_well_2,
                'path to well 3': path_to_well_3,
                'total path': path_total,
                'normalized path to well 1': None,
                'normalized path to well 2': None,
                'normalized path to well 3': None,
                'normalized total path': path_total/189.23
            })
            metric_indices.append(idx)
    else:
        for idx, row in trial_df.iterrows():
            start = row.get('start_frame_local', None)
            stop = row.get('stop_frame_local', None)
            global_start = row.get('start frame', None)
            reward_frame = safe_index(row.get('well_1_frame_local', None))
            
            if start is None or stop is None or pd.isna(start) or pd.isna(stop):
                print(f'[WARN] Trial {idx}: Missing local index — skipping.')
                continue

            start = int(start)
            stop = int(stop)
            global_start = int(global_start)

            end_cap = start + CAP
            end_stop = stop + 1
            end = min(end_cap, end_stop)

            if start < 0:
                start = 0
            if end > N:
                end = N

            if end - start <= 1:
                print(f'[WARN] Trial {idx}: end - start <= 1 -- Skipping Trial')

            path_total = extract.compute_path_length(x, y, start, stop)
            path_to_reward = (
                extract.compute_path_length(x, y, start, reward_frame)
                if reward_frame is not None else None
            )

            context = row.get('context',None)
            wc = 62.23 #cm
            bc = 99.06 #cm
            if context == 'wc':
                normalized_path_to_well = path_to_reward/wc if path_to_reward is not None else None
                normalized_total_path = path_total/(2*wc) if path_total is not None else None
            elif context == 'bc':
                normalized_path_to_well = path_to_reward/bc if path_to_reward is not None else None
                normalized_total_path = path_total/(2*bc) if path_total is not None else None
            else:
                normalized_path_to_well = None
                normalized_total_path = None

            head_dir,hd_frame = extract.get_trial_head_direction(hd, start, frame_offset=global_start)

            metrics_list.append({
                'end_frame_local':end,
                'head direction': head_dir,
                'hd frame': hd_frame,
                'path to well': path_to_reward,
                'total path': path_total,
                'normalized path to well': normalized_path_to_well,
                'normalized total path': normalized_total_path
            })
            metric_indices.append(idx)
    return pd.DataFrame(metrics_list, index=metric_indices)
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
def _moving_average(a, w=3, median_first=True):
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
def process_behavioral_data(dlc_data, center, arena_size=122, likelihood_threshold=0.2,
                            vel_threshold=10.0, spike_pad=1):
    import numpy as np

    # Pull arrays (dict or df supported if you reuse dlc_arrays)
    lear_x = np.asarray(dlc_data['lear.x'], dtype=float)
    lear_y = np.asarray(dlc_data['lear.y'], dtype=float)
    rear_x = np.asarray(dlc_data['rear.x'], dtype=float)
    rear_y = np.asarray(dlc_data['rear.y'], dtype=float)
    llk    = np.asarray(dlc_data['lear.score'], dtype=float)
    rlk    = np.asarray(dlc_data['rear.score'], dtype=float)

    Ls = {len(lear_x), len(lear_y), len(rear_x), len(rear_y), len(llk), len(rlk)}
    if len(Ls) != 1:
        raise ValueError(f"process_behavioral_data got unequal lengths: "
                         f"lear.x={len(lear_x)}, lear.y={len(lear_y)}, "
                         f"rear.x={len(rear_x)}, rear.y={len(rear_y)}, "
                         f"lear.score={len(llk)}, rear.score={len(rlk)}")

    # Likelihood masking
    lear_x[llk < likelihood_threshold] = np.nan
    lear_y[llk < likelihood_threshold] = np.nan
    rear_x[rlk < likelihood_threshold] = np.nan
    rear_y[rlk < likelihood_threshold] = np.nan

    with np.errstate(invalid='ignore'):
        x = np.nanmean(np.vstack([lear_x, rear_x]), axis=0)
        y = np.nanmean(np.vstack([lear_y, rear_y]), axis=0)

    # Scale
    x,y = scale_to_arena(x, y, arena_size, center=center)

    # Velocity-based spike cleanup
    vel = calculate_velocity(x, y)
    spike_idx = np.where(vel >= vel_threshold)[0]
    if spike_idx.size:
        if spike_pad > 0:
            pad = np.arange(-spike_pad, spike_pad + 1)
            spike_idx = np.unique(
                np.clip(spike_idx[:, None] + pad[None, :], 0, len(vel) - 1)
            )
        x[spike_idx] = np.nan; y[spike_idx] = np.nan
        lear_x[spike_idx] = np.nan; lear_y[spike_idx] = np.nan
        rear_x[spike_idx] = np.nan; rear_y[spike_idx] = np.nan

    dlcpos = np.column_stack((lear_x, lear_y, rear_x, rear_y))
    hd = calculate_head_direction(dlcpos)
    vel = calculate_velocity(x, y)
    return {'x': x, 'y': y, 'velocity': vel, 'head_direction': hd}
