import os
import sys
import pandas as pd
import numpy as np
from core import position_processing as pos
from core import trial_extraction as extract
from core import visualization as viz
from variants import curtain_context as cch

def run_pipeline(excel_path, experiment, summary):
    session_list = (pd.read_excel(excel_path)).to_dict('records')
    all_trial_dfs = []
    Data_Folder = os.path.join('/Volumes/ASA_LAB/Data/Julia/ATNRSC/experiments',f'{experiment}','data')

    CAPS = {'CH_3day': 30, 'CCH_40trial': 10, 'CCH_50trial': 10}
    cap_sec = CAPS.get(experiment, 15)


    for i, session in enumerate(session_list):
        print(f"\nProcessing Session {i+1}/{len(session_list)}")
        sfold = os.path.join(session['Path'],session['Session'])    # Session Folder path
        try:
            # ===== Section 1: Creating Trial Dataframe =====
            slp_path = os.path.join(sfold, session['Position Data'])
            score_path = os.path.join(sfold, session['Scoring File'])
            tracking_df = pd.read_csv(slp_path)
            score_df = pd.read_csv(score_path)
            center = (float(session['Center_x']),float(session['Center_y']))

            # analyze_trials_for_session extracts trial start and stop frames, context, trial time,
            # well visit frames, determines trial success, and builds trial dataframe
            trial_df = extract.analyze_session_trials(score_df, session, experiment)

            # ===== Section 2: Position Processing =====
            # Creates trimmed tracking data exclusive to trial frames
            tracking_trial_data = pos.slice_dlc_to_trials(tracking_df, trial_df)     
            frame_to_idx = {int(f): i for i, f in enumerate(tracking_trial_data['frame_idx'].to_numpy())}
            trial_df = extract.map_df(trial_df, experiment, frame_to_idx)
            arrs = pos.dlc_arrays(tracking_trial_data, cols=("lear.x","lear.y","rear.x","rear.y","lear.score","rear.score"))
            tracking_data = pos.process_behavioral_data(arrs,center)
            # Computing hd
            trial_df.update(pos.compute_trial_metrics(trial_df,experiment,tracking_data,cap_sec))
            
            # ===== Section 3: Visualization =====
            x, y = tracking_data['x'], tracking_data['y']
            session_label = f"{session['Rat']} {session['Session']}" # Title of Figure
            outdir = os.path.join(sfold,'plots')
            # Visualization Plots
            if session['Type'] == 'Block':
                viz.plot_collapsed_paths_by_blocks(
                    trial_df, session, x, y, outdir=outdir, max_blocks=5, use_local=True, session_label=session_label
                )
            elif session['Type'] == 'Ctx':
                viz.plot_collapsed_paths_by_context(trial_df, session, x, y, outdir=outdir, session_label=session_label)
            elif session['Type'] == 'Trial':
                for _, row in trial_df.iterrows():
                    trial = int(row.get('trial'))
                    viz.plot_trial_paths(
                        start = int(row.get('start_frame_local')),
                        end = int(row.get('end_frame_local')),
                        x = x,
                        y = y,
                        outfile = os.path.join(outdir,f'trial_{trial}.png'),
                        title = f'Trial {trial} trajectory',
                        button=False   #start/end points
                    )
            else:
                viz.plot_collapsed_trial_paths(trial_df,session,x,y,outdir=outdir,title=session_label)

            # ===== Section 4: Clean up and Saving =====
            
            # Clean up trial_df
            trial_df = trial_df.drop(['start_frame_local','stop_frame_local','end_frame_local'],axis=1)
            if experiment == 'CH_3day':
                trial_df = trial_df.drop(['well_1_frame_local','well_2_frame_local','well_3_frame_local'],axis=1)
            # Saving for Experiment Summarization
            all_trial_dfs.append(trial_df)

            # Save individual session results
            output_path = os.path.join(sfold,f'{session["Rat"]}_{session["Session"]}_trial_metrics.csv')
            trial_df.to_csv(output_path, float_format='%.2f',index=False)
            print(f'Session Trial Information saved to {output_path}')            

        except Exception as e:
            print(f'  Trial analysis failed: {e}') 

    # ===== Section 5: Experiment Summarization ===== UNDER CONSTRUCTION
    if summary:
        SESH_SUMMARY = os.path.join(Data_Folder,f'{experiment}_summary.csv')
        summary_df = cch.summarize_across_sessions(all_trial_dfs, session_list)
        summary_df.to_csv(SESH_SUMMARY, index=False)
        SUMMARY = os.path.join(Data_Folder,f'{experiment}_trials.csv')
        success_trials_df = pd.concat(all_trial_dfs)
        # success_trials_df = success_trials_df.drop(['session','start frame','stop frame','well visit frame','success',
        #                                 'head direction', 'wrong well visit', 'trial timeout'],axis=1)
        # success_trials_df = success_trials_df.groupby('day',as_index=False)
        success_trials_df.to_csv(SUMMARY, index=False) 

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python main.py /Path/To/RecList.xlsx ExperimentName --summary')
        sys.exit(1)

    excel_path = sys.argv[1]
    experiment = sys.argv[2]
    try:
        summary_opt = sys.argv[3]
    except:
        summary_opt = None

    if summary_opt is not None:
        summary = True
    else: 
        summary = False
    run_pipeline(excel_path,experiment, summary)

