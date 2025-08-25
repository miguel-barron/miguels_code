import os
import sys
import pandas as pd
import numpy as np
from core import data_io as d_up 
from core import position_processing as pos
from core import trial_extraction as extract
from core import visualization as viz
from variants import curtain_context as cch

# Define output folder
EXPERIMENT = '40trial_10day'
TRIAL_METRICS_FOLDER = f'output/{EXPERIMENT}/trial_metrics'
SUMMARY_CSV = f'output/{EXPERIMENT}/session_summary.csv'
os.makedirs(TRIAL_METRICS_FOLDER, exist_ok=True)

def run_pipeline(excel_path):
    session_list = d_up.load_session_metadata(excel_path)
    all_trial_dfs = []
    success_trials=[]
    idx = 1

    for i, session in enumerate(session_list):
        print(f"\nProcessing Session {i+1}/{len(session_list)}")

        try:
            slp_path = os.path.join(session['Path'], session['Position Data'])
            score_path = os.path.join(session['Path'], session['Scoring File'])
            
            tracking_dict = d_up.load_dlc_data(slp_path) # returns the tracking csv as a dictitionary
            score_df = d_up.load_score_file(score_path) # returns BORIS score as DataFrame

            trial_df = extract.analyze_trials_for_session(score_df) # Extracts trials and organizes trial information

            trial_df['rat'] = session['Rat']
            trial_df['session'] = session['Session']
            trial_df['day'] = session['Day']

            # Creates trimmed tracking data exclusive to trial frames
            tracking_trial_data = pos.slice_dlc_to_trials(tracking_dict, trial_df)
            frame_to_idx = {frame: i for i, frame in enumerate(tracking_trial_data.iloc[:,1])}
            trial_df = extract.map_df(trial_df,frame_to_idx)
            trial_df = cch.assign_context_blocks(trial_df)           
            tracking_data = pos.process_behavioral_data(d_up.df_to_dict(tracking_trial_data))

            # Computing x,y, head direction, and velocity
            trial_df.update(pos.compute_trial_metrics(trial_df,tracking_data))
            
            # Saving for Summarization and Plotting
            all_trial_dfs.append(trial_df)
            success_trials.append(trial_df[trial_df['success'] == True])

            # Visualization Prepping
            x, y = tracking_data['x'], tracking_data['y']
            session_label = f"{session['Rat']} {session['Session']}"
            outdir = os.path.join('output',EXPERIMENT, 'plots', f'{session['Rat']}_{session['Session']}')
            
            # Trial Visualization for Errors
            if session['Rat'] == 'GG08' and session['Session'] == '07222025_A':
                flagged = {25,27,28,29,30,41,42,43,45,47,48,49}
                for _, row in trial_df.iterrows():
                    trial = int(row.get('trial'))
                    if trial in flagged:
                        viz.plot_trial_paths(
                            start = int(row.get('start_frame_local')),
                            end = int(row.get('stop_frame_local')),
                            x = x,
                            y = y,
                            outfile = os.path.join('output','trial_plots',f'trial_{trial}.png'),
                            title = f'Trial {trial} trajectory'
                        )

            # Visualization Plots
            if session['Type'] == 'Block':
                viz.plot_collapsed_paths_by_blocks(
                    trial_df, x, y, outdir=outdir, block_size=10, max_blocks=5, use_local=True, session_label=session_label
                )
            if session['Type'] == 'Recall':
                viz.plot_collapsed_paths_by_context(
                    trial_df, x, y, outdir=outdir, session_label=session_label
                )
            else:
                for _, row in trial_df.iterrows():
                    trial = int(row.get('trial'))
                    viz.plot_trial_paths(
                        start = int(row.get('start_frame_local')),
                        end = int(row.get('stop_frame_local')),
                        x = x,
                        y = y,
                        outfile = os.path.join('output','trial_plots',f'trial_{trial}.png'),
                        title = f'Trial {trial} trajectory'
                    )

            # Save individual session results
            output_path = session.get('Trial Metrics Output', f'{session['Rat']}_trial_metrics_{session['Session']}.csv')
            full_output_path = os.path.join(TRIAL_METRICS_FOLDER, output_path)
            trial_df.to_csv(full_output_path, index=False)

        except Exception as e:
            print(f'  Trial analysis failed: {e}')

    # Summarize across sessions
    if all_trial_dfs:
        summary_df = cch.summarize_across_sessions(all_trial_dfs, session_list)
        summary_df.to_csv(SUMMARY_CSV, index=False)
        print(f'Session summary saved to: {SUMMARY_CSV}')

    if success_trials:
        success_trials_df = pd.concat(success_trials)
        success_trials_df = success_trials_df.drop(['session','start frame','stop frame','well visit frame','success',
                                     'head direction', 'wrong well visit', 'trial timeout','start_frame_local',
                                     'stop_frame_local','well_visit_local','idx_in_block'],axis=1)
        success_trials_df.groupby('day',as_index=False)
        success_trials_df.to_csv(f'output/{EXPERIMENT}/success_trials.csv', index=False)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python main.py /Path/To/RecList.xlsx')
        sys.exit(1)

    excel_path = sys.argv[1]
    run_pipeline(excel_path)