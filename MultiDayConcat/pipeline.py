from concat_utils import concatenate_day
from verify import verify_concatenation
import pandas as pd
import os
import shutil
import threading
import sys
# Attempt to import tqdm
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

DTYPE_NOTE = "# NOTE: Files are expected to contain raw int16 (2-byte) values"

successful_concatenation = []
failed_concatenation = []

def copy_folder_with_progress(src, dst):
    total_size = sum(os.path.getsize(os.path.join(dirpath, f))
                     for dirpath, _, filenames in os.walk(src)
                     for f in filenames)

    if tqdm:
        pbar = tqdm(total=total_size, unit="B", unit_scale=True, 
                    desc=f"Copying {os.path.basename(src)}")
    else: 
        print(f"Copying {os.path.basename(src)}...")
    
    def copy_func():
        if os.path.exists(dst):
            shutil.rmtree(dst)
        os.makedirs(dst, exist_ok=True)
        for dirpath, _, filenames in os.walk(src):
            rel_path = os.path.relpath(dirpath, src)
            dest_path = os.path.join(dst, rel_path)
            os.makedirs(dest_path, exist_ok=True)
            for file in filenames:
                src_file = os.path.join(dirpath, file)
                dst_file = os.path.join(dest_path, file)
                with open(src_file, 'rb') as fsrc, open(dst_file, 'wb') as fdst:
                    while chunk := fsrc.read(1024 * 1024):
                        fdst.write(chunk)
                        if tqdm:
                            pbar.update(len(chunk))
        if tqdm:
            pbar.close()

    thread = threading.Thread(target=copy_func)
    thread.start()
    return thread

def main(xlsx_path,temp_path,no_local=False):
    print("Starting Multi-Day Concatenation\n")
    if not no_local:
        os.makedirs(temp_dir, exist_ok=True)
    df = pd.read_excel(xlsx_path)

    copy_threads = []
    local_paths = {}

    for idx, row in df.iterrows():
        day = str(row['Day'])
        nas_path = row['Path']
        if not os.path.isdir(nas_path):
            print(f"Path not found: {nas_path}")
            continue

        local_day_path = nas_path if no_local else os.path.join(temp_path, os.path.basename(nas_path))
        local_paths[day] = (local_day_path, nas_path)
        if not no_local:
            thread = copy_folder_with_progress(nas_path, local_day_path)
            copy_threads.append(thread)

    for t in copy_threads:
        t.join()

    for day, (local_day_path, nas_path) in local_paths.items():
        print(f"\n=== {day} ===")

        result = concatenate_day(local_day_path)
        output_file = result['output_path']
        session_paths = result['session_paths']

        if verify_concatenation(output_file, session_paths):
            print(f"Verified: {output_file}")
            if no_local:
                successful_concatenation.append(local_day_path)
            else:
                shutil.copy2(output_file, os.path.join(nas_path, os.path.basename(output_file)))
                successful_concatenation.append(local_day_path)
        else:
            print(f"Validation failed: {output_file}")
            failed_concatenation.append(local_day_path)

    print("\n=== Summary ===")
    if failed_concatenation: 
        print("Successful Concatenations:")
        for path in successful_concatenation:
            print(f"  - {path}")
        print("Failed Concatenations:")
        for path in failed_concatenation:
            print(f"  - {path}")
        if not no_local:
            print("Cleaning up successful temp folders only...")
            for path in successful_concatenation:
                shutil.rmtree(path, ignore_errors=True)
    else:
        print("All days processed successfully and returned to NAS")
        if not no_local:
            print("Cleaning up all temp folders...")
            shutil.rmtree(temp_path, ignore_errors=True)

if __name__ == "__main__":
    args = sys.argv[1:]

    if len(args) < 2:
        print(f"Usage: python pipeline.py path_to_excel.xlsx path_to_temp_dir [--no-local]\n{DTYPE_NOTE}")
        sys.exit(1)

    excel_file = args[0]
    temp_dir = args[1]
    no_local = "--no-local" in args

    # Pass them to main()
    main(excel_file, temp_dir, no_local)
