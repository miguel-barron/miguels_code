# Multi-Day Concatenation Pipeline
This Python pipeline automates the process of concatenating binary `.dat` session files (e.g., `continuous_sess*.dat`) across multiple 
experimental days, with optional file copying and data verification. It is optimized for high-throughput environments and large binary 
files (e.g., 5 GB per session).

---
## Project Structure
An accompanying Excel file (`concat_days.xlsx`) should define the source mapping:

| Day     | SessionID | RatID | Experiment |                     DataPath                      |               OutputDir              | Order | Mode  |
|---------|-----------|-------|------------|---------------------------------------------------|--------------------------------------|-------|-------|
| 3312025 |     A     | AL02  | CH_2day    |/Volumes/.../sessions/03312025_A/.../continuous.dat|/Volumes/.../ALO2/concatEphys/03312025|   1   |source |
| 3312025 |     B     | AL02  | CH_2day    |/Volumes/.../sessions/03312025_B/.../continuous.dat|/Volumes/.../ALO2/concatEphys/03312025|   2   |source |
| 3312025 |     C     | AL02  | CH_2day    | ...                                               |...                                   |   3   |...    |

---
## Features

- **Binary-safe concatenation** of large `.dat` files using streaming I/O
- **NumPy-based verification** to validate file joins
- **Multi-threaded local copying** for performance optimization
- **Progress bars** via `tqdm` (optional)
- **Support for direct NAS operation** using the `--no-local` flag
- **Automatic cleanup** of local temp folders on success

---
## Pipeline Workflow

1. **Input**: Load Excel file with `Day` and `Path` columns.
2. **(Default)**: Copy each `Day {n}` folder locally for faster disk I/O. (optional)
3. **Concatenate** all `continuous_sess*.dat` files in numeric order using chunked binary I/O.
4. **Validate** concatenated output against original sessions using `np.array_equal()` with `int16` dtype.
5. **Return** validated file to the NAS (same Day folder).
6. **Cleanup**:
   - If **all days succeed**: Delete all local temp folders.
   - If **any mismatch occurs**: Retain failed folders for inspection.

---
## Usage

```bash Default Local Processing
python pipeline.py path_to_excel.xlsx path_to_temp_dir
```

With direct NAS access:

```bash
python pipeline.py path_to_excel.xlsx
```

---
## Requirements

- Python ≥ 3.8
- `pandas`, `numpy`
- `tqdm` (optional but recommended)

## Notes
- Files are assumed to be `int16` binary format (`dtype=np.int16`).

---
## Output
- Output files are saved as `concatenated_data.dat`, or `concatenated_data{n}.dat` if collisions occur.
- Results are logged to terminal with progress updates and summary of success/failure for each day.
