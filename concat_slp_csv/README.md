Script concats two SLEAP csv and and offsets frame_idx in the second file.

    Parameters:
    - csv1_path: Path to the first SLP .csv
    - csv2_path: Path to the second SLP .csv
    - output_path: Output path for the combined file
    - frame_offset: Total number of frames in the first video

    Usage: python concat_slp_csv.py csv1/path csv2/path output/path frame-offset-#