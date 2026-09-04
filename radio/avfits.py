import os
import sys
from pathlib import Path
from astropy.io import fits
import numpy as np


def average_fits_files(file_pattern, output_filename="averaged_output.fits"):
    # Split pattern into a base directory and a relative search pattern
    # This correctly parses complex paths like 'dir1/*/filename*.fits'
    parsed_path = Path(file_pattern)

    # Find where the wildcard starts to establish the base search directory
    parts = parsed_path.parts
    base_parts = []
    glob_parts = []
    found_wildcard = False

    for part in parts:
        if "*" in part or "?" in part:
            found_wildcard = True
        if found_wildcard:
            glob_parts.append(part)
        else:
            base_parts.append(part)

    # Default to current directory if pattern is just a filename
    base_dir = Path(*base_parts) if base_parts else Path(".")
    glob_pattern = os.path.join(*glob_parts) if glob_parts else ""

    # Use rglob if '**' is in pattern, otherwise use standard glob matching
    if "**" in glob_pattern:
        file_list = list(base_dir.rglob(glob_pattern.replace("**/", "")))
    else:
        file_list = list(base_dir.glob(glob_pattern))

    if not file_list:
        print(f"Error: No files found matching pattern '{file_pattern}'")
        return

    print(f"Found {len(file_list)} files to average...")

    # Load the first file to get shape and header data
    try:
        with fits.open(file_list[0]) as hdul:
            first_data = hdul[0].data[0]
            header = hdul[0].header[0]

        if first_data is None:
            print("Error: First FITS file contains no data array.")
            return

        # Initialize array to stack images
        all_data = np.zeros(
            (len(file_list), first_data.shape[0], first_data.shape[1], first_data.shape[2]),
            dtype=np.float64,
        )
        all_data[0] = first_data

    except Exception as e:
        print(f"Error reading first file {file_list[0]}: {e}")
        return

    # Load the rest of the files
    for i, file_path in enumerate(file_list[1:], start=1):
        try:
            with fits.open(file_path) as hdul:
                data = hdul[0].data[0] if hdul[0].data is not None else hdul[1].data[0]
                if data.shape != first_data.shape:
                    print(
                        f"Warning: Skipping {file_path} due to shape mismatch."
                    )
                    continue
                all_data[i] = data
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")

    # Compute the average
    print("Calculating average...")
    averaged_data = np.mean(all_data, axis=0)

    # Save the output
    try:
        primary_hdu = fits.PrimaryHDU(data=averaged_data)
        hdul_out = fits.HDUList([primary_hdu])
        hdul_out.writeto(output_filename, overwrite=True)
        print(f"Success! Averaged file saved as: {output_filename}")
    except Exception as e:
        print(f"Error saving output FITS file: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        pattern = sys.argv[1]
    else:
        pattern = input(
            "Enter the file name pattern (e.g., dir1/*/filename*.fits): "
        )

    average_fits_files(pattern)
