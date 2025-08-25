import pandas as pd
import numpy as np

from core.trial_metrics import build_trial_df
from variants import curtain_context as cch

def extract_trials(score_df):
    '''
    Extract trials from score file using 'START' and 'STOP' events.
    'START' and 'STOP' events are found in the 'Behavior type' column.
    Returns a list of tuples: (start_frame, stop_frame)
    '''
    trials = []
    start_idx = None

    for idx, row in score_df.iterrows():
        if row['Behavior type'] == 'START':
            if not pd.isna(row['Image index']):
                start_idx = int(row['Image index'])
            else:
                start_idx = None
        elif row['Behavior type'] == 'STOP' and start_idx is not None:
            end_idx = row['Image index']
            if not pd.isna(end_idx):
                trials.append((start_idx, int(end_idx)))
            start_idx = None  # Reset either way

    return trials
# ------------------------------------------------
def analyze_trials_for_session(score_df):
    trials = extract_trials(score_df)

    context_map = score_df.set_index('Image index')['Behavior'].to_dict()
    trial_summaries = []

    for i, (start, end) in enumerate(trials):
        try:
            context = context_map.get(start, 'Unknown')
            reward_frame = get_reward_frame(score_df, start, end)
            trial_num = i+1
            trial_info = build_trial_df(
                trial_num, start, end, reward_frame, context
            )
            if trial_info is None:
                print(f'     Trial {trial_num} failed: trial_info is None (likely due to out-of-bounds)')
                continue  # Skip further processing for this trial
            trial_time = trial_info['trial time']
            trial_info['success'], trial_info['wrong well visit'], trial_info['trial timeout'] = cch.determine_success_custom(
                                                                                            score_df, start, end, reward_frame,trial_time)
            trial_summaries.append(trial_info)

        except Exception as e:
            print(f'     Trial {trial_num} failed: {e}')

    df = pd.DataFrame(trial_summaries)
    return df
# ------------------------------------------------
def get_reward_frame(trial_df, start, stop):
    '''
    Get frame number for correct well visit ('1') between trial START and STOP.
    '''
    well_visit = trial_df[(trial_df['Image index'] >= start) & (trial_df['Image index'] <= stop)]
    well_visit = well_visit[well_visit['Behavior'] == '1']
    if not well_visit.empty:
        return int(well_visit.iloc[0]['Image index'])
    return None
# ------------------------------------------------
def get_wrong_well_frame(trial_df, start, stop):
    '''
    Get frame number for wrong well visit ('2') between trial START and STOP.
    '''
    wrong_rows = trial_df[(trial_df['Image index'] >= start) & (trial_df['Image index'] <= stop)]
    wrong_rows = wrong_rows[wrong_rows['Behavior'] == '2']
    if not wrong_rows.empty:
        return int(wrong_rows.iloc[0]['Image index'])
    return None
# ------------------------------------------------
def get_timeout_frame(trial_df, start, stop):
    '''
    Get frame number for trial timeout ('x') between START and STOP.
    '''
    timeout_rows = trial_df[(trial_df['Image index'] >= start) & (trial_df['Image index'] <= stop)]
    timeout_rows = timeout_rows[trial_df['Behavior'] == 'x']
    if not timeout_rows.empty:
        return int(timeout_rows.iloc[0]['Image index'])
    return None
# ------------------------------------------------
def get_trial_context(trial_df, start, stop):
    '''
    Determine the context label for the trial based on 'wc' or 'bc' within window.
    '''
    window = trial_df[(trial_df['Image index'] >= start) & (trial_df['Image index'] <= stop)]
    ctx = window[window['Behavior'].isin(['wc', 'bc'])]['Behavior'].values
    if len(ctx) > 0:
        return ctx[0]
    return 'unknown'
# ------------------------------------------------
def get_trial_head_direction(hd_array, start_frame):
    '''
    Return head direction at start frame (if valid).
    '''
    if 0 <= start_frame < len(hd_array):
        return hd_array[start_frame]
    return np.nan
# ------------------------------------------------
def get_trial_path(x, y, start, end):
    '''
    Slice x and y position arrays from start to end frame.
    Returns two arrays of same length.
    '''
    x_slice = x[start:end]
    y_slice = y[start:end]
    return x_slice, y_slice
# ------------------------------------------------
def compute_path_length(x, y, start, end):
    x_slice = x[start:end]
    y_slice = y[start:end]
    if len(x_slice) != len(y_slice):
        print(f'[ERROR] x and y length mismatch: x={len(x_slice)}, y={len(y_slice)}')
    if np.all(np.isnan(x_slice)) or np.all(np.isnan(y_slice)):
        print(f'[WARN] All NaNs in trial slice: start={start}, end={end}')
        return None

    dx = np.diff(x_slice)
    dy = np.diff(y_slice)
    dist = np.sqrt(dx**2 + dy**2)
    return np.nansum(dist)
# ------------------------------------------------
def find_event_between(trial_df, label, start_frame, end_frame):
    for idx, row in trial_df.iterrows():
        if (
            row['Behavior'] == label
            and start_frame <= row['Image index'] <= end_frame
        ):
            return int(row['Image index'])
    return None
# ------------------------------------------------
def map_df(trial_df,frame_to_idx):
    trial_df['start_frame_local'] = pd.to_numeric(trial_df['start frame'].map(frame_to_idx),errors='coerce')
    trial_df['stop_frame_local'] = pd.to_numeric(trial_df['stop frame'].map(frame_to_idx),errors='coerce')
    trial_df['well_visit_local'] = pd.to_numeric(trial_df['well visit frame'].map(frame_to_idx),errors='coerce')
    return trial_df
# ------------------------------------------------
def collect_metrics(trial_df):
    df = trial_df.copy()
    df_new = df[df['success']==True]