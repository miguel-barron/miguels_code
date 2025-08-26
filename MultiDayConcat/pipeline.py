from concat_utils import *
from verify import verify_concatenation
import pandas as pd                                                         #type: ignore
import os
import shutil
import sys
# Attempt to import tqdm
try:
    from tqdm import tqdm                                                   #type: ignore
except ImportError:
    tqdm = None

CHUNK_SIZE = 1024 * 1024 * 4 # 4 MB 
DTYPE_NOTE = "# NOTE: Files are expected to contain raw int16 (2-byte) values"

def main(xlsx_path, temp_path='', local=False, stage_only=False):
    print("Starting Multi-Day Concatenation\n")

    if local:
        os.makedirs(temp_path, exist_ok=True)

    df = pd.read_excel(xlsx_path)

    successful_concatenation = []
    mismatch_concatenation = []

    for idx, row in df.iterrows():
        idx +=1
        day = str(row["Day"])
        nas_path = str(row["Path"])

        if not os.path.isdir(nas_path):
            print(f"[WARN] Path not found: {nas_path}")
            mismatch_concatenation.append(day)
            continue

        # One Day → one local working dir (when copying locally)
        local_day_path = nas_path if not local else os.path.join(temp_path, f"Day {idx}")

        try:
            # --- 1) COPY (strictly sequential) ---
            if local:
                print(f"[COPY] Day {idx}: {nas_path} → {local_day_path}")
                t = copy_folder_with_progress(nas_path, local_day_path)
                t.join()  # <= makes the copy blocking (sequential)
                print("[COPY] Clone complete.\n")
            else:
                print(f"[INFO] Day {idx}: working in-place at {local_day_path}\n")

            # --- 2) CONCATENATE ---
            concat_info = concatenate_day(local_day_path)
            output_path = concat_info["output_path"]
            session_paths = [str(p) for p in concat_info["session_paths"]]

            # --- 3) VERIFY (with tqdm) ---
            print("[VERIFY] Streaming comparison...")
            ok = verify_concatenation(output_path, session_paths)
            if not ok:
                print(f"[ERROR] Verification failed for Day {day}. Keeping working folder: {local_day_path}")
                mismatch_concatenation.append(day)
                continue

            # --- 4) RETURN RESULT TO NAS ---
            final_dst = os.path.join(nas_path, os.path.basename(output_path))
            if os.path.exists(final_dst):
                stem, ext = os.path.splitext(final_dst)
                k = 1
                while os.path.exists(f"{stem}{k}{ext}"):
                    k += 1
                final_dst = f"{stem}{k}{ext}"

            if stage_only:
                print(f"[STAGE-ONLY] Keeping local concatenated file for Day {idx}: {output_path}")
            else:
                print(f"[RETURN] Uploading to source destination: {final_dst}")
                return_with_progress(output_path, final_dst, remove_src=True)
                print(f"[RETURN] Complete: {final_dst}")

            successful_concatenation.append(final_dst)

            # --- 5) CLEANUP (only if we copied locally) ---
            if local:
                if not stage_only:
                    try:
                        print(f"[CLEAN] Removing temp: {local_day_path}")
                        shutil.rmtree(local_day_path, ignore_errors=True)
                        print(f"[CLEAN] Removed temp: {local_day_path}")
                    except Exception as e:
                        print(f"[WARN] Could not remove temp '{local_day_path}': {e}")

        except Exception as e:
            print(f"[ERROR] Day {idx}: {e}")
            mismatch_concatenation.append(day)

    # --- SUMMARY ---
    print("\n=== Summary ===")
    if not mismatch_concatenation:
        print(f'Concatenation Successful for all entries')
    else:
        print(f"  Success: {len(successful_concatenation)}")
        for p in successful_concatenation:
            print(f"    → {p}")
        print(f"  Failed : {len(mismatch_concatenation)}")
        print(f"    Days: {mismatch_concatenation}")

if __name__ == "__main__":
    args = sys.argv[1:]

    if len(args) > 1: 
        print('Starting Temp Process')
        excel_file = args[0]
        temp_dir = args[1]
        if temp_dir:
            local = True
        stage_only = "--stage-only" in args

        
        main(excel_file, temp_dir, local, stage_only)
    elif len(args) == 1:
        print('Starting Source Process')
        excel_file = args[0]

        main(excel_file)
    else: 
        print(f"Usage: python pipeline.py path_to_excel.xlsx path_to_temp_dir [--stage-only]\n{DTYPE_NOTE}")
        sys.exit(1)