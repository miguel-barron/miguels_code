from concat_utils import concatenate_day
from verify import verify_concatenation
import pandas as pd                                                         #type: ignore
import os
import shutil
import threading
import sys
import time
# Attempt to import tqdm
try:
    from tqdm import tqdm                                                   #type: ignore
except ImportError:
    tqdm = None

CHUNK_SIZE = 1024 * 1024 * 4 # 4 MB 
DTYPE_NOTE = "# NOTE: Files are expected to contain raw int16 (2-byte) values"

successful_concatenation = []
failed_concatenation = []

def _is_hidden_or_junk(path):
    name = os.path.basename(path)
    if name.startswith("."):  # .DS_Store, ._foo
        return True
    # Extra guard for common macOS trash
    return name in ("Icon\r", "VolumeIcon.icns")


def copy_folder_with_progress(src, dst):
    # Collect all files & total size safely
    total_size = 0
    file_list = []
    for dirpath, _, filenames in os.walk(src):
        for f in filenames:
            src_file = os.path.join(dirpath, f)
            if os.path.islink(src_file) or f.startswith("."):   # skip symlinks & hidden junk
                continue
            try:
                size = os.path.getsize(src_file)
            except OSError:
                continue  # skip unreadable/missing files
            rel_path = os.path.relpath(dirpath, src)
            dst_file = os.path.join(dst, rel_path, f)
            file_list.append((src_file, dst_file, size))
            total_size += size

    def copy_func():
        if os.path.exists(dst):
            shutil.rmtree(dst)
        os.makedirs(dst, exist_ok=True)

        with tqdm(total=total_size, unit="B", unit_scale=True, desc=f"Copying {os.path.basename(src)}") as pbar:
            for src_file, dst_file, size in file_list:
                os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                with open(src_file, "rb") as fsrc, open(dst_file, "wb") as fdst:
                    while chunk := fsrc.read(CHUNK_SIZE):
                        fdst.write(chunk)
                        pbar.update(len(chunk))

    thread = threading.Thread(target=copy_func)
    thread.start()
    return thread

def main(xlsx_path, temp_path, no_local=False):
    print("Starting Multi-Day Concatenation\n")

    if not no_local:
        os.makedirs(temp_path, exist_ok=True)

    df = pd.read_excel(xlsx_path)

    successful_concatenation = []
    mismatch_concatenation = []

    for _, row in df.iterrows():
        day = str(row["Day"])
        nas_path = str(row["Path"])

        if not os.path.isdir(nas_path):
            print(f"[WARN] Path not found: {nas_path}")
            mismatch_concatenation.append(day)
            continue

        # One Day → one local working dir (when copying locally)
        local_day_path = nas_path if no_local else os.path.join(temp_path, f"Day {day}")

        try:
            # --- 1) COPY (strictly sequential) ---
            if not no_local:
                print(f"[COPY] Day {day}: {nas_path} → {local_day_path}")
                t = copy_folder_with_progress(nas_path, local_day_path)
                t.join()  # <= makes the copy blocking (sequential)
                print("[COPY] Clone complete.\n")
            else:
                print(f"[INFO] Day {day}: working in-place at {local_day_path}\n")

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
            shutil.move(output_path, final_dst)
            print(f"[RETURN] {final_dst}")

            successful_concatenation.append(final_dst)

            # --- 5) CLEANUP (only if we copied locally) ---
            if not no_local:
                try:
                    shutil.rmtree(local_day_path, ignore_errors=True)
                    print(f"[CLEAN] Removed temp: {local_day_path}\n")
                except Exception as e:
                    print(f"[WARN] Could not remove temp '{local_day_path}': {e}\n")

        except Exception as e:
            print(f"[ERROR] Day {day}: {e}")
            mismatch_concatenation.append(day)

    # --- SUMMARY ---
    print("\n=== Summary ===")
    print(f"  Success: {len(successful_concatenation)}")
    for p in successful_concatenation:
        print(f"    → {p}")
    print(f"  Failed : {len(mismatch_concatenation)}")
    if mismatch_concatenation:
        print(f"    Days: {mismatch_concatenation}")

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
