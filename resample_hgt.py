#!/usr/bin/env python3
'''Resample SRTM1 (3601x3601) HGT zip files to SRTM3 (1201x1201).
Takes every 3rd pixel in both axes.'''

import argparse
import glob
import math
import os
import struct
import sys
import zipfile
from multiprocessing import Pool


def resample_file(args):
    '''Resample one HGT file from SRTM1 to SRTM3.'''
    inpath, outpath, overwrite = args
    basename = os.path.basename(inpath)
    try:
        if os.path.exists(outpath) and not overwrite:
            return (basename, "skipped")

        with zipfile.ZipFile(inpath, 'r') as zf:
            inner_name = zf.namelist()[0]
            data = zf.read(inner_name)

        size = int(math.sqrt(len(data) // 2))
        if size != 3601:
            return (basename, "not SRTM1 (size=%d)" % size)

        # Parse as big-endian int16 rows, take every 3rd pixel
        row_bytes = size * 2
        out_rows = []
        for row in range(0, size, 3):
            offset = row * row_bytes
            row_data = struct.unpack('>%dh' % size, data[offset:offset + row_bytes])
            out_rows.append(struct.pack('>%dh' % 1201, *row_data[::3]))

        out_data = b''.join(out_rows)

        # Write as zip with same inner name
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        tmp_path = outpath + '.tmp'
        with zipfile.ZipFile(tmp_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(inner_name, out_data)
        os.rename(tmp_path, outpath)

        return (basename, "ok")
    except Exception as e:
        return (basename, "error: %s" % e)


def main():
    parser = argparse.ArgumentParser(
        description='Resample SRTM1 HGT files to SRTM3')
    parser.add_argument('input_dir', help='Directory containing SRTM1 .hgt.zip files')
    parser.add_argument('output_dir', help='Output directory for SRTM3 .hgt.zip files')
    parser.add_argument('--processes', type=int, default=8, help='Number of parallel workers')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files')
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, '**/*.hgt.zip'), recursive=True))
    if not files:
        print("No .hgt.zip files found in %s" % args.input_dir)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    work = []
    for f in files:
        basename = os.path.basename(f)
        outpath = os.path.join(args.output_dir, basename)
        work.append((f, outpath, args.overwrite))

    print("Resampling %d files with %d processes..." % (len(work), args.processes))

    errors = 0
    done = 0
    skipped = 0
    with Pool(processes=args.processes) as pool:
        for basename, status in pool.imap_unordered(resample_file, work, chunksize=16):
            if status == "ok":
                done += 1
            elif status == "skipped":
                skipped += 1
            else:
                errors += 1
                print("  %s: %s" % (basename, status))
            if (done + skipped + errors) % 1000 == 0:
                print("  %d / %d processed" % (done + skipped + errors, len(work)))

    print("Done: %d resampled, %d skipped, %d errors" % (done, skipped, errors))


if __name__ == '__main__':
    main()
