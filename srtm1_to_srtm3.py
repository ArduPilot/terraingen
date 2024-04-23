#!/usr/bin/env python3
'''
resample SRTM1 data to SRTM3
'''

import os
import zipfile
import numpy as np
import rasterio
import argparse
from scipy import interpolate
from concurrent.futures import ThreadPoolExecutor

def resample_data(data, new_shape):
    """
    Resample the data array to the new shape using mean pooling.

    Parameters:
        data (numpy.array): The original data array.
        new_shape (tuple): The shape of the new resampled array.

    Returns:
        numpy.array: The resampled data array.
    """
    old_height, old_width = data.shape
    new_height, new_width = new_shape

    x = np.linspace(0, 1, data.shape[0])
    y = np.linspace(0, 1, data.shape[1])
    f = interpolate.interp2d(y, x, data, kind='cubic')

    x2 = np.linspace(0, 1, new_width)
    y2 = np.linspace(0, 1, new_height)
    resampled = f(y2, x2)

    return resampled.astype(data.dtype)

def process_hgt_file(filepath, output_folder):
    with zipfile.ZipFile(filepath, 'r') as z:
        z.extractall(output_folder)

    filename = os.path.basename(filepath).replace('.zip', '')
    file_path = os.path.join(output_folder, filename)

    tmp_out = f"SRTM3_tmp_{filename}"
    if not os.path.exists(file_path):
        print("Expected %s to exist - skipping" % file_path)
        return
    with rasterio.open(file_path) as src:
        data = src.read(1)
        new_data = resample_data(data, (1201, 1201))  # SRTM3 resolution is 1201x1201 for 3 arc seconds
        new_data = new_data.byteswap(inplace=True)

        new_file_path = os.path.join(output_folder, tmp_out)
        new_data.tofile(new_file_path)

    os.remove(file_path)  # Clean up the extracted file

    with zipfile.ZipFile(new_file_path + '.zip', 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.write(new_file_path, arcname=filename)
    
    os.remove(new_file_path)  # Clean up the uncompressed new HGT file
    os.rename(os.path.join(output_folder,tmp_out + ".zip"), file_path + ".zip")

    print(f"Processed and resampled {filename} to SRTM3 resolution.")

def process_directory(input_folder, output_folder, max_workers, overwrite):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for filename in os.listdir(input_folder):
            if filename.endswith(".hgt.zip"):
                if not overwrite and os.path.exists(os.path.join(output_folder, filename)):
                    print("Skipping existing file %s" % os.path.join(output_folder, filename))
                    continue
                futures.append(executor.submit(process_hgt_file, os.path.join(input_folder, filename), output_folder))
        for future in futures:
            future.result()

def main():
    parser = argparse.ArgumentParser(description="Convert SRTM1 HGT files to SRTM3 resolution with optional parallel processing.")
    parser.add_argument("input_folder", type=str, help="Directory containing SRTM1 HGT files")
    parser.add_argument("output_folder", type=str, help="Directory to save converted SRTM3 files")
    parser.add_argument("--parallel", type=int, default=1, help="Specify the number of threads to use for parallel processing")
    parser.add_argument("--overwrite", action='store_true', help="overwrite existing files")
    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    process_directory(args.input_folder, args.output_folder, args.parallel, args.overwrite)

if __name__ == "__main__":
    main()
