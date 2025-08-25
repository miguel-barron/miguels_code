import numpy as np
import pandas as pd

def find_event_between(trial_df, label, start_frame, end_frame):
    for idx, row in trial_df.iterrows():
        if (
            row['Behavior'] == label
            and start_frame <= row['Image index'] <= end_frame
        ):
            return int(row['Image index'])
    return None

def determine_success_custom(score_df, start_frame, end_frame, reward_frame,trial_time):
    wrong_well = find_event_between(score_df, '2', start_frame, end_frame)
    timeout = find_event_between(score_df, 'x', start_frame, end_frame)
    
    wrong_well_visit = False
    trial_timeout = False
    
    if reward_frame is None:
        success = False
    elif wrong_well:
        success = False
        wrong_well_visit = True
    elif timeout or trial_time > 10:
        success = False
        trial_timeout = True
    else:
        success = True
    return success, wrong_well_visit, trial_timeout
# ------------------------------------------------
def assign_context_blocks(trial_df):
    df = trial_df.copy().sort_values('start_frame_local').reset_index(drop=True)
    # keep only wc/bc trials for plotting
    df = df[df['context'].isin(['wc', 'bc'])].copy()

    # new block whenever context changes
    df['block'] = (df['context'] != df['context'].shift()).cumsum()

    # optional: index within block
    df['idx_in_block'] = df.groupby('block').cumcount() + 1
    return df
# ------------------------------------------------
def summarize_blocks_across_sessions(trial_dfs, session_list):
    all_blocks = []
    for df, session in zip(trial_dfs, session_list):
        df = df.copy()

        # Filter out invalid context labels "1" and "2"
        df = df[~df['context'].isin(['1', '2'])]

        # Defensive check
        if df.empty or 'context' not in df.columns:
            print(f'[WARN] Skipping session {session['Session']} for rat {session['Rat']} — no valid trials or missing context column.')
            continue

        # Sort by trial start frame
        df = df.sort_values('start frame').reset_index(drop=True)

        block_data = []
        current_context = None
        block_trials = []
        block_index = 1

        for _, row in df.iterrows():
            if row['context'] != current_context:
                if block_trials:
                    block_df = pd.DataFrame(block_trials)
                    block_data.append({
                        'rat': session['Rat'],
                        'session': session['Session'],
                        'block': block_index,
                        'context': current_context,
                        'trial_start': int(block_df['trial'].iloc[0]),
                        'trial_end': int(block_df['trial'].iloc[-1]),
                        'n_trials': len(block_df),
                        'n_success': block_df['success'].sum(),
                        'success_rate': block_df['success'].mean()
                    })
                    block_index += 1
                current_context = row['context']
                block_trials = [row]
            else:
                block_trials.append(row)

        if block_trials:
            block_df = pd.DataFrame(block_trials)
            block_data.append({
                'rat': session['Rat'],
                'session': session['Session'],
                'block': block_index,
                'context': current_context,
                'trial_start': int(block_df['trial'].iloc[0]),
                'trial_end': int(block_df['trial'].iloc[-1]),
                'n_trials': len(block_df),
                'n_success': block_df['success'].sum(),
                'success_rate': block_df['success'].mean()
            })

        all_blocks.append(pd.DataFrame(block_data))

    return pd.concat(all_blocks, ignore_index=True) if all_blocks else pd.DataFrame()
# ------------------------------------------------
def summarize_across_sessions(trial_dfs, session_list):
    rows = []

    for df, sess in zip(trial_dfs, session_list):
        if df is None or df.empty or 'context' not in df.columns:
            print(f'[WARN] Skipping session {sess.get('Session','?')} for rat {sess.get('Rat','?')} — no valid trials or missing context.')
            continue

        # keep only wc/bc and sort by trial order
        use = df[df['context'].isin(['wc', 'bc'])].copy()
        if use.empty:
            continue
        # pick a sort key you trust; falling back to 'trial' if present
        sort_key = 'start_frame_local' if 'start_frame_local' in use.columns else (
            'start_frame' if 'start_frame' in use.columns else 'trial'
        )
        use = use.sort_values(sort_key).reset_index(drop=True)

        sess_type = sess.get('Type', 'Block')  # default learning
        rat_id    = sess.get('Rat', '')
        sess_id   = sess.get('Session', '')

        if sess_type == 'Recall':
            g = use.groupby('context', as_index=False).agg(
                n_trials=('trial', 'count'),
                n_success=('success', 'sum')
            )
            g['success_rate'] = g['n_success'] / g['n_trials']
            for _, r in g.iterrows():
                rows.append({
                    'summary_type': 'recall_context',
                    'rat': rat_id,
                    'session': sess_id,
                    'block': np.nan,
                    'context': r['context'],
                    'n_trials': int(r['n_trials']),
                    'n_success': int(r['n_success']),
                    'success_rate': float(r['success_rate']),
                })
        else:
            # Learning days: contiguous blocks by context run-length
            current_context = None
            block_trials = []
            block_index = 0

            def flush_block(bt, ctx, bi):
                if not bt:
                    return
                bdf = pd.DataFrame(bt)
                rows.append({
                    'summary_type': 'learning_block',
                    'rat': rat_id,
                    'session': sess_id,
                    'block': bi,
                    'context': ctx,
                    'n_trials': int(len(bdf)),
                    'n_success': int(bdf['success'].sum()),
                    'success_rate': float(bdf['success'].mean()) if len(bdf) else np.nan,
                })

            for _, row in use.iterrows():
                if row['context'] != current_context:
                    flush_block(block_trials, current_context, block_index)
                    current_context = row['context']
                    block_trials = [row]
                    # only advance block index once we *finish* the block
                    if block_index == 1 and current_context is None:
                        pass
                    else:
                        # when we flushed, we already used block_index; increment now
                        block_index += 1
                else:
                    block_trials.append(row)

            # flush last block
            flush_block(block_trials, current_context, block_index)

    if not rows:
        return pd.DataFrame(columns=['summary_type','rat','session','block','context','n_trials','n_success','success_rate'])

    out = pd.DataFrame(rows, columns=['summary_type','rat','session','block','context','n_trials','n_success','success_rate'])
    # Optional tidy: sort
    out = out.sort_values(['rat','session','summary_type','block','context'], na_position='last').reset_index(drop=True)
    return out