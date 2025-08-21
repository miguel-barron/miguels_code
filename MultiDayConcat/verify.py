import os

CHUNK_SIZE = 1024 * 1024  # 1 MB

def verify_concatenation(output_path, session_paths):
    """
    Efficiently verify that the concatenated output file contains
    all session files in exact byte order using chunked streaming.
    """
    try:
        with open(output_path, "rb") as out_f:
            for session_file in session_paths:
                with open(session_file, "rb") as sess_f:
                    while True:
                        sess_chunk = sess_f.read(CHUNK_SIZE)
                        out_chunk = out_f.read(len(sess_chunk))
                        
                        if not sess_chunk:
                            break
                        if sess_chunk != out_chunk:
                            print(f"Mismatch detected in: {session_file}")
                            return False
        return True
    except Exception as e:
        print(f"Error during verification: {e}")
        return False