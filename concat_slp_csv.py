import pandas as pd
import sys

def concat(csv1_path, csv2_path, output_path, frame_offset):
    """
    Concatenates two SLEAP CSVs and offsets frame_idx in the second file.
    
    Parameters:
    - csv1_path: Path to the first SLP .csv
    - csv2_path: Path to the second SLP .csv
    - output_path: Output path for the combined file
    - frame_offset: Total number of frames in the first video
    """
    # Load CSVs
    df1 = pd.read_csv(csv1_path)
    df2 = pd.read_csv(csv2_path)

    # Offset frame indices in second file
    df2 = df2.copy()
    df2["frame_idx"] += frame_offset

    # Concatenate
    df_combined = pd.concat([df1, df2], ignore_index=True)

    # Save output
    df_combined.to_csv(output_path, index=False)
    print(f"Combined file saved as: {output_path}")

if __name__ == "__main__":
    args = sys.argv[1:]

    if len(args) == 4: 
        csv1_path = str(args[0])
        csv2_path = str(args[1])
        output_path = str(args[2])
        frame_offset = int(args[3])
        concat(csv1_path, csv2_path, output_path, frame_offset)
    else: 
        print("Usage: python concat_slp_csv.py csv1/path csv2/path output/path frame-offset-#")
        sys.exit(1)
