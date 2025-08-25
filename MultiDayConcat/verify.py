import os
from tqdm import tqdm                                                         #type: ignore

CHUNK_SIZE = 1024 * 1024  # 1 MB

def verify_concatenation(output_path, session_paths):
    """
    Verify that the concatenated output file matches all session files in order.
    Uses tqdm to show a single aggregated progress bar.
    Returns True if everything matches, False otherwise.
    """
    try:
        # Total size of all session files
        total_size = sum(
            os.path.getsize(p) for p in session_paths if os.path.exists(p)
        )

        with open(output_path, "rb") as out_f, tqdm(
            total=total_size, unit="B", unit_scale=True, desc="Verifying"
        ) as pbar:
            for session_file in session_paths:
                with open(session_file, "rb") as sess_f:
                    while True:
                        sess_chunk = sess_f.read(CHUNK_SIZE)
                        if not sess_chunk:
                            break
                        out_chunk = out_f.read(len(sess_chunk))
                        if sess_chunk != out_chunk:
                            print(f"\n[VERIFY] Mismatch detected in {session_file}")
                            return False
                        pbar.update(len(sess_chunk))
        return True
    except Exception as e:
        print(f"[VERIFY] Error: {e}")
        return False