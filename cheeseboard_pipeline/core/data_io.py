import pandas as pd

# ========== Data Loaders ==========

def load_dlc_data(dlc_path):
    df = pd.read_csv(dlc_path)
    return df

def df_to_dict(dlc_df):
    df = dlc_df
    return {
        'frame_idx':df.iloc[:,1].to_numpy,
        'lear_x': df.iloc[:,3].to_numpy(),
        'lear_y': df.iloc[:,4].to_numpy(),
        'lear_lk': df.iloc[:,5].to_numpy(),
        'rear_x': df.iloc[:,6].to_numpy(),
        'rear_y': df.iloc[:,7].to_numpy(),
        'rear_lk': df.iloc[:,8].to_numpy()
        }
def load_score_file(score_path):
    return pd.read_csv(score_path)

def load_session_metadata(excel_path):
    df = pd.read_excel(excel_path)
    return df.to_dict("records")