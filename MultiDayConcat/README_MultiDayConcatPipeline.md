# Multi-Day Concatenation Pipeline

This Python pipeline automates the process of concatenating binary `.dat` session files (e.g., `continuous_sess*.dat`) across multiple experimental days, with optional file copying and data verification. It is optimized for high-throughput environments and large binary files (e.g., 5 GB per session).

---

## Project Structure

Each experimental day is expected to follow this format:

```
Day {n}/
├── continuous_sess1.dat
├── continuous_sess2.dat
└── continuous_sess3.dat
```

An accompanying Excel file (`.xlsx`) should define the source mapping:

| Day     | Path                          |
|---------|-------------------------------|
| Day 1   | /Volumes/.../Data/Day 1       |
| Day 2   | /Volumes/.../Data/Day 2       |
| ...     | ...                           |

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
python pipeline.py ~/Downloads/concat_days.xlsx ~/Temp/ConcatPipeline
```

With direct NAS access:

```bash
python pipeline.py ~/Downloads/concat_days.xlsx ~/Data/Julia/ATNRSC --no-local
```

---

## Requirements

- Python ≥ 3.8
- `pandas`, `numpy`
- `tqdm` (optional but recommended)

To install dependencies:

```bash
pip install pandas numpy tqdm
```

---

## Notes

- Files are assumed to be `int16` binary format (`dtype=np.int16`).
- Session files must be named using the convention: `continuous_sess{n}.dat`.

---

## Output

- Output files are saved as `concatenated_data.dat`, or `concatenated_data{n}.dat` if collisions occur.
- Results are logged to terminal with progress updates and summary of success/failure for each day.

---

## Recommended Use

- Use on local systems with fast SSD storage for preprocessing.
- Avoid running on Wi-Fi connected machines due to slower transfer rates.
- Ideal for labs working with high-resolution ephys data needing integrity-verified batch preprocessing.

---