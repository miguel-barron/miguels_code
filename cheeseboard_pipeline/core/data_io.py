import pandas as pd
import numpy as np

# ========== Data Frame to Dictionary for SLP.csv ==========
def df_to_dict(dlc_df):
    df = dlc_df
    return {
        'frame_idx':df.iloc[:,1].to_numpy(),
        'lear.x': df.iloc[:,3].to_numpy(),
        'lear.y': df.iloc[:,4].to_numpy(),
        'lear.score': df.iloc[:,5].to_numpy(),
        'rear.x': df.iloc[:,6].to_numpy(),
        'rear.y': df.iloc[:,7].to_numpy(),
        'rear.score': df.iloc[:,8].to_numpy()
        }

def validate_trial_indices(trial_df, x_len, sf_col='start_frame_local', ef_col='end_frame_local'):
    bad = []
    for i, row in trial_df.iterrows():
        s = row.get(sf_col, None)
        e = row.get(ef_col, None)
        if pd.isna(s): s = 0
        if pd.isna(e): e = x_len
        try:
            s = int(s); e = int(e)
        except Exception:
            bad.append((i, s, e, "non-int"))
            continue
        if not (0 <= s < e <= x_len):
            bad.append((i, s, e, "oob/order"))
    return bad

def valid_local_trials(trial_df, x_len,
                       sf="start_frame_local", ef="end_frame_local"):
    """Return a boolean mask of trials that have valid, in-bounds local indices."""
    ok = pd.Series(True, index=trial_df.index)
    for col in (sf, ef):
        ok &= trial_df[col].notna()
    # cast safely to int for comparisons, but only where ok==True
    s = trial_df.loc[ok, sf].astype("int64")
    e = trial_df.loc[ok, ef].astype("int64")
    ok.loc[ok] &= (s >= 0) & (e <= x_len) & (e > s + 1)
    return ok

def debug_scale_report(x_raw, y_raw, center, arena_size, label=""):
    import numpy as np

    def stats(a, name):
        f = np.isfinite(a)
        if not f.any():
            return f"{name}: all-NaN"
        return (f"{name}: n={f.sum()} "
                f"min={np.nanmin(a):.3f} med={np.nanmedian(a):.3f} "
                f"max={np.nanmax(a):.3f}")

    print("\n[DEBUG scale report]", label)
    print(" center (pixels):", center)

    # Pre-center summaries
    print(" raw:", stats(x_raw, "x_raw"), "|", stats(y_raw, "y_raw"))

    # Center and flip (same as your function)
    x_c = x_raw.astype(float) - center[0]
    y_c = y_raw.astype(float) - center[1]
    y_c = -y_c  # assuming flip_y=True

    f = np.isfinite(x_c) & np.isfinite(y_c)
    if not f.any():
        print(" centered: all-NaN")
        return

    x_f, y_f = x_c[f], y_c[f]
    x_lo, x_hi = np.quantile(x_f, [0.01, 0.99])
    y_lo, y_hi = np.quantile(y_f, [0.01, 0.99])
    ext_x = max(x_hi - x_lo, 0.0)
    ext_y = max(y_hi - y_lo, 0.0)

    print(f" centered: x_lo={x_lo:.3f} x_hi={x_hi:.3f} ext_x={ext_x:.3f} "
          f"| y_lo={y_lo:.3f} y_hi={y_hi:.3f} ext_y={ext_y:.3f}")

    if ext_x <= 1e-6 or ext_y <= 1e-6:
        print("tiny/zero extent on one axis → scale collapse risk")

    sx = arena_size / (ext_x if ext_x > 1e-6 else 1.0)
    sy = arena_size / (ext_y if ext_y > 1e-6 else 1.0)
    print(f" scale_x={sx:.6f} scale_y={sy:.6f}")

    x_scaled = x_c * sx + (arena_size/2.0)
    y_scaled = y_c * sy + (arena_size/2.0)

    print(" scaled:", stats(x_scaled, "x_cm"), "|", stats(y_scaled, "y_cm"))


def build_frame_index(df, frame_col="frame_idx"):
    """Return {global_frame:int -> local_index:int} with NaNs/dups filtered out."""
    vals = df[frame_col].to_numpy()
    finite = np.isfinite(vals)
    vals = vals[finite].astype("int64", copy=False)

    # ensure strictly increasing (if your normalized df guarantees this, great)
    # if duplicates exist, keep the first occurrence
    frame_to_idx = {}
    for i, f in enumerate(vals):
        if f not in frame_to_idx:
            frame_to_idx[f] = i
    return frame_to_idx