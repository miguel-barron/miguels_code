import pandas as pd
import numpy as np

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
def analyze_session_trials(score_df, session, experiment):
    trials = extract_trials(score_df)

    context_map = score_df.set_index('Image index')['Behavior'].to_dict()
    trial_summaries = []

    if experiment == 'CH_3day':
        for i, (start, stop) in enumerate(trials):
            try:
                trial_num = i+1
                # Ensure start and end frames are valid integers and not NaN
                if np.isnan(start) or np.isnan(stop):
                    print(f'Trial {trial_num}: NaN in start or end frame — skipping.')
                    continue

                start = int(start)
                stop = int(stop)

                trial_time = float((stop - start) / 30)
                trial_timeout = trial_time > 30

                context = context_map.get(start, 'Unknown')
                well_1_frame = get_well_frame(score_df, start, stop, well = 1)
                well_1_time = float((int(well_1_frame) - start) / 30) if well_1_frame else None
                well_2_frame = get_well_frame(score_df, start, stop, well = 2)
                well_2_time = float((int(well_2_frame) - start) / 30) if well_2_frame else None
                well_3_frame = get_well_frame(score_df, start, stop, well = 3)
                well_3_time = float((int(well_3_frame) - start) / 30) if well_3_frame else None

                # Determine Success
                success = all([well_1_frame, well_2_frame, well_3_frame]) and not trial_timeout

                trial_info = {
                                'day': session['Day'],
                                'rat': session['Rat'],
                                'session': session['Session'],
                                'trial': trial_num,
                                'context': context,
                                'head direction': None,
                                'hd frame': None,
                                'start frame': start,
                                'well 1 frame': well_1_frame,
                                'well 2 frame': well_2_frame,
                                'well 3 frame': well_3_frame,
                                'stop frame': stop,
                                'end_frame_local': None,
                                'success': success,
                                'trial time': trial_time,
                                'well 1 time': well_1_time,
                                'well 2 time': well_2_time,
                                'well 3 time': well_3_time,
                                'path to well 1': None,
                                'path to well 2': None,
                                'path to well 3': None,
                                'total path': None,
                                'normalized path to well 1': None,
                                'normalized path to well 2': None,
                                'normalized path to well 3': None,
                                'normalized total path': None,
                                'trial timeout': trial_timeout
                            }

                trial_summaries.append(trial_info)

            except Exception as e:
                print(f'     Trial {trial_num} failed: {e}')
            
    else:
        for i, (start, stop) in enumerate(trials):
            try:
                trial_num = i+1
                # Ensure start and end frames are valid integers and not NaN
                if np.isnan(start) or np.isnan(stop):
                    print(f'Trial {trial_num}: NaN in start or end frame — skipping.')
                    continue

                start = int(start)
                stop = int(stop)

                trial_time = (stop - start) / 30
                trial_timeout = trial_time > 10

                context = context_map.get(start, 'Unknown')
                reward_frame = get_reward_frame(score_df, start, stop)
                reward_time = (int(reward_frame) - start) / 30 if reward_frame else None
                well_2_frame = get_well_frame(score_df, start, stop, well = 2)
                
                wrong_well = well_2_frame is not None

                # Determine Success
                if not reward_frame or trial_timeout or wrong_well:
                    success = False
                else:
                    success = True

                trial_info = {
                                'day': session['Day'],
                                'rat': session['Rat'],
                                'session': session['Session'],
                                'trial': trial_num,
                                'context': context,
                                'head direction': None,
                                'hd frame': None,
                                'start frame': start,
                                'well 1 frame': reward_frame,
                                'well 1 time': reward_time,
                                'stop frame': stop,
                                'end_frame_local': None,
                                'success': success,
                                'trial time': trial_time,
                                'path to well': None,
                                'total path': None,
                                'normalized path to well': None,
                                'normalized total path': None,
                                'trial timeout': trial_timeout,
                                'wrong well visit': wrong_well
                            }

                trial_summaries.append(trial_info)

            except Exception as e:
                print(f'     Trial {trial_num} failed: {e}')

    return pd.DataFrame(trial_summaries) # To be saved as
# ------------------------------------------------
def get_well_frame(score_df, start, stop, well):
    '''
    Get frame number for correct well visit ('i') between trial START and STOP.
    '''
    well = str(well)
    well_visit = score_df[(score_df['Image index'] >= start) & (score_df['Image index'] <= stop)]
    well_visit = well_visit[well_visit['Behavior'] == well ]
    if not well_visit.empty:
        return int(well_visit.iloc[0]['Image index'])
    return None
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
def get_trial_head_direction(hd_array, start_frame, frame_offset=0, max_lookahead=15):
    """
    Return the first valid head direction at or after `start_frame`, 
    adjusted to a global frame index if `frame_offset` is provided.

    Parameters
    ----------
    hd_array : np.ndarray
        Array of head directions (local or global).
    start_frame : int
        Start index within hd_array.
    frame_offset : int, optional
        Offset to convert local frame indices to global frame indices.
        Default = 0 means array is already global.
    max_lookahead : int, optional
        Max number of frames to search forward.

    Returns
    -------
    (hd_value, global_index) : tuple
    """
    if start_frame < 0 or start_frame >= len(hd_array):
        return np.nan, None

    end_frame = len(hd_array) if max_lookahead is None else min(start_frame + max_lookahead, len(hd_array))
    sub = hd_array[start_frame:end_frame]
    valid_idx = np.where(~np.isnan(sub))[0]
    if len(valid_idx) == 0:
        return np.nan, None

    first_valid_offset = valid_idx[0]
    hd_value = sub[first_valid_offset]
    hd_frame_global = first_valid_offset + frame_offset

    return hd_value, hd_frame_global
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
def map_df(trial_df,experiment,frame_to_idx):
    if experiment == 'CH_3day':
        trial_df['start_frame_local'] = pd.to_numeric(trial_df['start frame'].map(frame_to_idx),errors='coerce')
        trial_df['stop_frame_local'] = pd.to_numeric(trial_df['stop frame'].map(frame_to_idx),errors='coerce')
        trial_df['well_1_frame_local'] = pd.to_numeric(trial_df['well 1 frame'].map(frame_to_idx),errors='coerce')
        trial_df['well_2_frame_local'] = pd.to_numeric(trial_df['well 2 frame'].map(frame_to_idx),errors='coerce')
        trial_df['well_3_frame_local'] = pd.to_numeric(trial_df['well 3 frame'].map(frame_to_idx),errors='coerce')
    else:
        trial_df['start_frame_local'] = pd.to_numeric(trial_df['start frame'].map(frame_to_idx),errors='coerce')
        trial_df['stop_frame_local'] = pd.to_numeric(trial_df['stop frame'].map(frame_to_idx),errors='coerce')
        trial_df['well_1_frame_local'] = pd.to_numeric(trial_df['well 1 frame'].map(frame_to_idx),errors='coerce')
    return trial_df
# ------------------------------------------------
def map_frames_safe(trial_df, frame_to_idx,
                    cols=("start_frame", "stop_frame", "well_visit")):
    """Create *_local as pandas nullable Int64; do NOT cast to Python int yet."""
    trial_df = trial_df.copy()
    for c in cols:
        if c in trial_df.columns:
            m = trial_df[c].map(frame_to_idx)  # returns floats/NaN if missing
            # convert to pandas nullable integer (keeps <NA> instead of bombing)
            trial_df[f"{c}_local"] = m.astype("Int64")
    return trial_df

# def get_well_visits(score_df, start, stop, well='1'):
#     '''
#     Get frame number for well visit ('1') between trial START and STOP.
#     '''
#     well_visit = score_df[(score_df['Image index'] >= start) & (score_df['Image index'] <= stop)]
#     well_visit = well_visit[well_visit['Behavior'] == well]
#     if not well_visit.empty:
#         if 
#         return int(well_visit.iloc[0]['Image index'])
#     return None

# def analyze_trials_for_FE(score_df):
#     trials = extract_trials(score_df)

#     context_map = score_df.set_index('Image index')['Behavior'].to_dict()
#     trial_summaries = []

#     for i, (start, end) in enumerate(trials):
#         try:
#             reward_frame = get_reward_frame(score_df, start, end)
#             trial_num = i+1
#             trial_info = build_trial_df(
#                 trial_num, start, end, reward_frame
#             )
#             if trial_info is None:
#                 print(f'     Trial {trial_num} failed: trial_info is None (likely due to out-of-bounds)')
#                 continue  # Skip further processing for this trial
#             trial_time = trial_info['trial time']
#             trial_info['success'], trial_info['wrong well visit'], trial_info['trial timeout'] = cch.determine_success_custom(
#                                                                                             score_df, start, end, reward_frame,trial_time)
#             trial_summaries.append(trial_info)

#         except Exception as e:
#             print(f'     Trial {trial_num} failed: {e}')

#     df = pd.DataFrame(trial_summaries)
#     return df