from pathlib import Path
import os
import shutil
import pandas as pd
import sys
from typing import List, Dict

from concat_utils import concatenate_day  # provided by user
from verify import verify_concatenation   # provided by user

DTYPE_NOTE = "# NOTE: Files are expected to contain raw int16 (2-byte) values"
CHUNK_SIZE = 1024 * 1024 * 4  # 4MB copy chunks


def copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Source path does not exist: {src}")
    dst.mkdir(parents=True, exist_ok=True)

    for root, dirs, files in os.walk(src):
        rel = os.path.relpath(root, src)
        target_dir = dst / rel if rel != "." else dst
        target_dir.mkdir(parents=True, exist_ok=True)

        for f in files:
            s = Path(root) / f
            d = target_dir / f
            with open(s, "rb") as rf, open(d, "wb") as wf:
                while True:
                    chunk = rf.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    wf.write(chunk)
            try:
                shutil.copystat(s, d, follow_symlinks=False)
            except Exception:
                pass  # best-effort metadata


def move_file(src_file: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_file = dst_dir / src_file.name
    if dst_file.exists():
        stem = src_file.stem
        suffix = src_file.suffix
        k = 1
        while True:
            candidate = dst_dir / f"{stem}{k}{suffix}"
            if not candidate.exists():
                dst_file = candidate
                break
            k += 1
    shutil.move(str(src_file), str(dst_file))
    return dst_file


def process_day(day_label: str, day_src_path: Path, temp_root: Path, no_local: bool) -> Dict:
    print(f"\\n=== Processing Day: {day_label} ===")
    if no_local:
        working_path = day_src_path
        print(f"Working in-place: {working_path}")
    else:
        working_path = temp_root / f"Day {day_label}"
        print(f"Copying to local temp: {working_path}")
        if working_path.exists():
            print(f"[WARN] Temp dir already exists, reusing: {working_path}")
        else:
            copy_tree(day_src_path, working_path)

    concat_info = concatenate_day(str(working_path))
    out_path = Path(concat_info["output_path"])
    session_paths = concat_info["session_paths"]

    print("Verifying concatenation by streaming compare...")
    ok = verify_concatenation(str(out_path), [str(p) for p in session_paths])
    if not ok:
        print(f"[ERROR] Verification failed for Day {day_label}. Keeping temp for inspection: {working_path}")
        return {"day": day_label, "ok": False, "working_path": str(working_path), "output": str(out_path)}

    print("Verification passed. Returning concatenated file to NAS...")
    final_dst = move_file(out_path, day_src_path)
    print(f"→ Returned to: {final_dst}")

    if not no_local:
        print("Cleaning up local temp for this day...")
        try:
            shutil.rmtree(working_path, ignore_errors=True)
        except Exception as e:
            print(f"[WARN] Failed to remove temp folder: {e}")

    return {"day": day_label, "ok": True, "final_path": str(final_dst)}


def main(excel_file: str, temp_dir: str, no_local: bool = False) -> int:
    df = pd.read_excel(excel_file)
    cols = {c.lower(): c for c in df.columns}
    if "day" not in cols or "path" not in cols:
        raise ValueError("Excel file must contain columns: 'Day' and 'Path'")

    temp_root = Path(temp_dir)
    temp_root.mkdir(parents=True, exist_ok=True)

    successes: List[Dict] = []
    failures: List[Dict] = []

    for _, row in df.iterrows():
        day_label = str(row[cols["day"]])
        day_src_path = Path(str(row[cols["path"]])).expanduser()

        try:
            result = process_day(day_label, day_src_path, temp_root, no_local)
            if result.get("ok"):
                successes.append(result)
            else:
                failures.append(result)
        except Exception as e:
            print(f"[ERROR] Unhandled exception on Day {day_label}: {e}")
            failures.append({"day": day_label, "ok": False, "error": str(e)})

    print("\\n=== Summary ===")
    print(f"Successful days: {len(successes)}")
    for s in successes:
        print(f"  Day {s['day']} → {s.get('final_path', s.get('output'))}")
    print(f"Failed days: {len(failures)}")
    for f in failures:
        print(f"  Day {f['day']} → {f.get('error', 'verification failed')}")

    return 0 if len(failures) == 0 else 1


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 2:
        print(f"Usage: python pipeline_sequential.py path_to_excel.xlsx path_to_temp_dir [--no-local]\\n{DTYPE_NOTE}")
        sys.exit(1)

    excel_file = args[0]
    temp_dir = args[1]
    no_local = "--no-local" in args

    rc = main(excel_file, temp_dir, no_local)
    sys.exit(rc)
