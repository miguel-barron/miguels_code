import os
import re
try:
    from tqdm import tqdm                                                         #type: ignore
except ImportError:
    tqdm = None

CHUNK_SIZE = 1024 * 1024 * 4 # 4 MB 
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
                        while chunk := infile.read(CHUNK_SIZE):
                            outfile.write(chunk)
                            pbar.update(len(chunk))
                else:
                    # fallback to simple printing
                    total_read = 0
                    while chunk := infile.read(CHUNK_SIZE):
                        outfile.write(chunk)
                        total_read += len(chunk)
                    print(f"      → {total_read / (1024**2):.2f} MB written.")

    print(f"Finished writing to: {output_file}")

    return {
        "output_path": output_file,
        "session_paths": session_paths
    }