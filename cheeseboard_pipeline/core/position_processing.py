import numpy as np # type: ignore
import pandas as pd # type: ignore
from scipy.ndimage import uniform_filter1d # type: ignore
import core.trial_extraction as extract

def slice_dlc_to_trials(dlc_df, trial_df):
    # Step 1: Build full set of frame indices across all trials
    trial_ranges = set()
    for _, row in trial_df.iterrows():
        start = int(row['start frame'])
        end = int(row['stop frame'])
        trial_ranges.update(range(start, end + 1))
    
    # Step 2: Find missing frame indices
    dlc_frame_set = set(dlc_df.iloc[:, 1].astype(int))  # Assume column 1 is frame_idx
    missing_frames = sorted(list(trial_ranges - dlc_frame_set))

    if missing_frames:
        print(f'[INFO] Inserting {len(missing_frames)} missing frame rows into DLC...')
        # Step 3: Create dummy rows with NaNs
        missing_df = pd.DataFrame({
            dlc_df.columns[1]: missing_frames  # frame_idx column
        })
        # Fill all other columns with NaN
        for col in dlc_df.columns:
            if col != dlc_df.columns[1]:
                missing_df[col] = np.nan

        # Step 4: Concatenate and sort by frame_idx
        df = pd.concat([dlc_df, missing_df], ignore_index=True)
        df = df.sort_values(by=dlc_df.columns[1]).reset_index(drop=True)
    else:
        df = dlc_df.copy()

    # Step 5: Apply slicing
    df_trimmed = df[df.iloc[:, 1].isin(trial_ranges)].reset_index(drop=True)
    return df_trimmed
# ------------------------------------------------
def scale_to_arena(x, y, arena_size, center=(675,619), flip_y=True):
    '''
    Scale raw coordinates to arena coordinates in cm, center them,
    and optionally flip the y-axis to match physical orientation.

    Parameters
    ----------
    x, y : np.ndarray
        Raw coordinate arrays.
    arena_size : float
        Arena diameter in cm.
    center : tuple
        Pixel coordinates (center_x, center_y) of the arena center in raw coords.
    flip_y : bool
        Whether to flip the y-axis.
    '''

    rad = arena_size/2
    # Translate so that arena center is at (0, 0)
    x_centered = x - center[0]
    y_centered = y - center[1]

    # Optionally flip y-axis
    if flip_y:
        y_centered = -y_centered

    # Find scaling factor based on arena diameter in pixels
    max_dist_x = np.nanmax(x_centered) - np.nanmin(x_centered)
    max_dist_y = np.nanmax(y_centered) - np.nanmin(y_centered)

    scale_x = arena_size / max_dist_x
    scale_y = arena_size / max_dist_y

    # Apply scaling
    x_scaled = x_centered * scale_x
    y_scaled = y_centered * scale_y

    # Recenter at Arena Center
    x_scaled_centered = x_scaled + rad
    y_scaled_centered = y_scaled + rad
    return x_scaled_centered, y_scaled_centered
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
def compute_trial_metrics(trial_df, tracking_data):
    x, y, hd = tracking_data['x'], tracking_data['y'], tracking_data['head_direction']
    metrics_list = []
    metric_indices = []

    for idx, row in trial_df.iterrows():
        start = (int(row['start_frame_local']) if not pd.isna(row['start_frame_local']) else None)
        end = (int(row['stop_frame_local']) if not pd.isna(row['stop_frame_local']) else None)
        reward_frame = (int(row['well_visit_local']) if not pd.isna(row['well_visit_local']) else None)

        if start is None or end is None:
            print(f'[WARN] Trial {idx}: Missing local index — skipping.')
            continue

        path_total = extract.compute_path_length(x, y, start, end)
        path_to_reward = (
            extract.compute_path_length(x, y, start, reward_frame)
            if reward_frame is not None else None
        )
        head_dir = extract.get_trial_head_direction(hd, start)

        metrics_list.append({
            'head direction': head_dir,
            'path to well': path_to_reward,
            'total path': path_total
        })
        metric_indices.append(idx)
    result = pd.DataFrame(metrics_list, index=metric_indices)
    return result
# ------------------------------------------------
def interpolate_nans(arr):
    nans = np.isnan(arr)
    if np.any(~nans):
        arr[nans] = np.interp(
            np.flatnonzero(nans),
            np.flatnonzero(~nans),
            arr[~nans]
        )
    return arr
# ------------------------------------------------
def get_column_array(data, key):
    if isinstance(data, pd.DataFrame):
        return data[key].to_numpy()
    elif isinstance(data, dict):
        return np.asarray(data[key])
    else:
        raise TypeError(f"Unsupported data type for key '{key}': {type(data)}")
# ------------------------------------------------
def process_behavioral_data(dlc_data, arena_size=122, likelihood_threshold= 0.3):
    lear_x = get_column_array(dlc_data,'lear_x')
    rear_x = get_column_array(dlc_data,'rear_x')
    lear_y = get_column_array(dlc_data,'lear_y')
    rear_y = get_column_array(dlc_data,'rear_y')
    llk = get_column_array(dlc_data,'lear_lk')
    rlk = get_column_array(dlc_data,'rear_lk')

    lear_x[llk < likelihood_threshold] = np.nan
    lear_y[llk < likelihood_threshold] = np.nan
    rear_x[rlk < likelihood_threshold] = np.nan
    rear_y[rlk < likelihood_threshold] = np.nan

    with np.errstate(invalid='ignore'):
        x = np.nanmean(np.vstack([lear_x, rear_x]), axis=0)
        y = np.nanmean(np.vstack([lear_y, rear_y]), axis=0)

    x = interpolate_nans(x)
    y = interpolate_nans(y)
    x = remove_outliers(x)
    y = remove_outliers(y)
    x, y = scale_to_arena(x, y, arena_size)

    # x_smooth, y_smooth = smooth_positions(x, y)
    # vel = calculate_velocity(x_smooth, y_smooth)
    vel = calculate_velocity(x,y)
    vel[vel > 100] = np.nan
    dlcpos = np.column_stack((lear_x, lear_y, rear_x, rear_y))
    hd = calculate_head_direction(dlcpos)
    return {
        'x': x,
        'y': y,
        'velocity': vel,
        'head_direction': hd
    }