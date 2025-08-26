import os
import re
import threading
import shutil

try:
    from tqdm import tqdm                                                         #type: ignore
except ImportError:
    tqdm = None

CHUNK = 1024 * 1024 * 4 # 4 MB 
DTYPE = "int16" # for context and validation elsewhere

def extract_session_index(filename):
    match = re.search(r"continuous_sess(\d+)\.dat", filename)
    return int(match.group(1)) if match else float('inf')

def get_next_output_filename(day_path):
    base_name = "concatenated_data"
    existing = [f for f in os.listdir(day_path) if f.startswith(base_name) and f.endswith(".dat")]

    if f"{base_name}.dat" not in existing:
        return os.path.join(day_path, f"{base_name}.dat")

    suffixes = [int(re.search(r"(\d+)", f).group(1)) for f in existing if re.search(r"concatenated_data(\d+).dat", f)]
    next_index = max(suffixes, default=1) + 1
    return os.path.join(day_path, f"{base_name}{next_index}.dat")

def return_with_progress(src_file: str, dst_file: str, remove_src: bool = True) -> str:
    """
    Copy src_file -> dst_file over SMB with tqdm progress and resume support.
    If dst_file already exists and is smaller than src, appends from the last byte.
    Returns the final destination path.
    """
    os.makedirs(os.path.dirname(dst_file), exist_ok=True)

    src_size = os.path.getsize(src_file)
    # Determine resume point if a partial exists
    start = 0
    try:
        if os.path.exists(dst_file):
            dst_size = os.path.getsize(dst_file)
            if 0 <= dst_size < src_size:
                start = dst_size
            elif dst_size >= src_size:
                # Destination already complete or larger; overwrite fresh
                os.remove(dst_file)
    except OSError:
        # If stat fails on NAS, start fresh
        start = 0

    mode = "ab" if start else "wb"

    with open(src_file, "rb") as rf, open(dst_file, mode) as wf, tqdm(
        total=src_size, initial=start, unit="B", unit_scale=True,
        desc=f"Uploading {os.path.basename(src_file)}"
    ) as pbar:
        if start:
            rf.seek(start)
        while True:
            chunk = rf.read(CHUNK)
            if not chunk:
                break
            wf.write(chunk)
            pbar.update(len(chunk))

    # Extra flush to be safe on network FS
    try:
        wf.flush()  # type: ignore
        os.fsync(wf.fileno())  # type: ignore
    except Exception:
        pass

    if remove_src:
        try:
            os.remove(src_file)
        except Exception:
            pass

    return dst_file

def concatenate_day(day_path):
    """
    Concatenate all continuous_sess*.dat files in a Day folder into one output file.
    Returns a dict with output file path and list of session paths.
    """
    files = [f for f in os.listdir(day_path) if f.startswith("continuous_sess") and f.endswith(".dat")]
    files_sorted = sorted(files, key=extract_session_index)

    output_file = get_next_output_filename(day_path)
    session_paths = []

    print(f"Processing Day folder: {day_path}")
    print(f"Output will be saved as: {os.path.basename(output_file)}")

    with open(output_file, "wb") as outfile:
        for fname in files_sorted:
            fpath = os.path.join(day_path, fname)
            fsize = os.path.getsize(fpath)
            session_paths.append(fpath)
            print(f"   Concatenating: {fname}")

            with open(fpath, "rb") as infile:
                if tqdm:
                    with tqdm(
                        total=fsize, unit='B', unit_scale=True, unit_divisor=1024,
                        desc=fname, ncols=80, leave=False
                    ) as pbar:
                        while chunk := infile.read(CHUNK):
                            outfile.write(chunk)
                            pbar.update(len(chunk))
                else:
                    # fallback to simple printing
                    total_read = 0
                    while chunk := infile.read(CHUNK):
                        outfile.write(chunk)
                        total_read += len(chunk)
                    print(f"      → {total_read / (1024**2):.2f} MB written.")

    print(f"Finished writing to: {output_file}")

    return {
        "output_path": output_file,
        "session_paths": session_paths
    }

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
            for src_file, dst_file, __ in file_list:
                os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                with open(src_file, "rb") as fsrc, open(dst_file, "wb") as fdst:
                    while chunk := fsrc.read(CHUNK):
                        fdst.write(chunk)
                        pbar.update(len(chunk))

    thread = threading.Thread(target=copy_func)
    thread.start()
    return thread