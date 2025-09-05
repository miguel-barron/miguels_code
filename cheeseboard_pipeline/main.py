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
EXPERIMENT = 'CCH_50trial_10day'
Data_Folder = os.path.join('/Volumes/ASA_LAB/Data/Julia/ATNRSC/experiments',f'{EXPERIMENT}','data')

def run_pipeline(excel_path):
    session_list = d_up.load_session_metadata(excel_path)
    all_trial_dfs = []
    success_trials=[]
    idx = 1

    for i, session in enumerate(session_list):
        print(f"\nProcessing Session {i+1}/{len(session_list)}")
        sfold = session['Path']
        vtype = str(session['Version'])
        try:
            slp_path = os.path.join(sfold, session['Position Data'])
            score_path = os.path.join(sfold, session['Scoring File'])
            
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

            # Visualization Prepping
            x, y = tracking_data['x'], tracking_data['y']
            session_label = f"{session['Rat']} {session['Session']}"
            outdir = os.path.join(sfold, 'plots',vtype)

            # Visualization Plots
            if session['Type'] == 'Block':
                viz.plot_collapsed_paths_by_blocks(
                    trial_df, x, y, outdir=outdir, max_blocks=5, use_local=True, session_label=session_label
                )
            if session['Type'] == 'Ctx':
                viz.plot_collapsed_paths_by_context(
                    trial_df, x, y, outdir=outdir, session_label=session_label
                )
            if session['Type'] == 'Trial':
                for _, row in trial_df.iterrows():
                    trial = int(row.get('trial'))
                    viz.plot_trial_paths(
                        start = int(row.get('start_frame_local')),
                        end = int(row.get('end_frame_local')),
                        x = x,
                        y = y,
                        outfile = os.path.join(outdir,f'trial_{trial}.png'),
                        title = f'Trial {trial} trajectory'
                    )
            # Clean up trial_df
            trial_df = trial_df.drop(['start_frame_local','stop_frame_local','well_visit_local','idx_in_block'],axis=1)

            # Saving for Summarization and Plotting
            # all_trial_dfs.append(trial_df)
            # success_trials.append(trial_df[trial_df['success'] == True])

            # Save individual session results
            # output_path = os.path.join(sfold,'trial_metrics.csv')
            # trial_df.to_csv(output_path, index=False)

        except Exception as e:
            print(f'  Trial analysis failed: {e}')

    # # Summarize across sessions
    # if all_trial_dfs:
    #     SUMMARY = os.path.join(Data_Folder,'session_summary.csv')
    #     summary_df = cch.summarize_across_sessions(all_trial_dfs, session_list)
    #     summary_df.to_csv(SUMMARY, index=False)

    # if success_trials:
    #     SUCCESS = os.path.join(Data_Folder,'successful_trials.csv')
    #     success_trials_df = pd.concat(success_trials)
    #     success_trials_df = success_trials_df.drop(['session','start frame','stop frame','well visit frame','success',
    #                                  'head direction', 'wrong well visit', 'trial timeout'],axis=1)
    #     success_trials_df.groupby('day',as_index=False)
    #     success_trials_df.to_csv(SUCCESS, index=False)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python main.py /Path/To/RecList.xlsx')
        sys.exit(1)

    excel_path = sys.argv[1]
    run_pipeline(excel_path)