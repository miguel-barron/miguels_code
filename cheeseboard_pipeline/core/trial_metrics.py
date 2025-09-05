import numpy as np

# Build Trial DataFrame
def build_trial_df(trial_num, start, end, reward_frame, context):
    try:
        # Ensure start and end frames are valid integers and not NaN
        if np.isnan(start) or np.isnan(end):
            print(f'Trial {trial_num}: NaN in start or end frame — skipping.')
            return None

        start = int(start)
        end = int(end)

        trial_time = (end - start + 1) / 30

        # Reward frame can be None or NaN
        if reward_frame is not None and not np.isnan(reward_frame):
            reward_frame = int(reward_frame)
        else:
            reward_frame = None

        success = None

        return {
            'day': None,
            'rat': None,
            'session': None,
            'trial': trial_num,
            'context': context,
            'start frame': start,
            'well visit frame': reward_frame,
            'stop frame': end,
            'end_frame_local': None,
            'success': success,
            'head direction': None,
            'trial time': trial_time,
            'path to well': None,
            'total path': None,
            'wrong well visit': None,
            'trial timeout': None
        }

    except Exception as e:
        print(f'Trial {trial_num}: Failed to build dataframe — {e}')
        return None