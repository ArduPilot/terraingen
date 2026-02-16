#!/usr/bin/env python3
'''Find terrain files with steep height differences between neighbouring points.'''

import argparse
import glob
import gzip
import math
import os
import re
import struct
import sys
import zipfile
from multiprocessing import Pool

import numpy as np

IO_BLOCK_SIZE = 2048
TERRAIN_GRID_BLOCK_SIZE_X = 28
TERRAIN_GRID_BLOCK_SIZE_Y = 32

TILE_RE = re.compile(r'([NS])(\d{2})([EW])(\d{3})')


def file_type(filepath):
    name = filepath.lower()
    if name.endswith('.hgt.zip'):
        return 'hgt'
    if name.endswith('.dat.gz') or name.endswith('.dat'):
        return 'dat'
    return None


def parse_tile_coords(filepath):
    m = TILE_RE.search(os.path.basename(filepath))
    if m is None:
        return None
    lat = int(m.group(2))
    lon = int(m.group(4))
    if m.group(1) == 'S':
        lat = -lat
    if m.group(3) == 'W':
        lon = -lon
    return (lat, lon)


def scan_hgt(args):
    filepath, threshold = args
    basename = os.path.basename(filepath)
    try:
        coords = parse_tile_coords(filepath)
        tile_lat, tile_lon = coords if coords else (0, 0)

        with zipfile.ZipFile(filepath, 'r') as zf:
            data = zf.read(zf.namelist()[0])
        size = int(math.sqrt(len(data) // 2))
        n = size - 1
        arr = np.frombuffer(data, dtype='>i2').reshape(size, size).astype(np.float32)
        # Mark voids as NaN so they don't trigger false positives
        arr[arr == -32768] = np.nan

        # Check horizontal and vertical neighbours
        diff_h = np.abs(np.diff(arr, axis=1))
        diff_v = np.abs(np.diff(arr, axis=0))

        steep_h = np.nansum(diff_h > threshold)
        steep_v = np.nansum(diff_v > threshold)

        # Find location of max difference
        max_h = np.nanmax(diff_h) if diff_h.size > 0 else 0
        max_v = np.nanmax(diff_v) if diff_v.size > 0 else 0
        if max_h >= max_v and max_h > 0:
            idx = np.nanargmax(diff_h)
            row, col = divmod(idx, diff_h.shape[1])
        elif max_v > 0:
            idx = np.nanargmax(diff_v)
            row, col = divmod(idx, diff_v.shape[1])
        else:
            row, col = 0, 0
        max_diff = max(max_h, max_v)

        # HGT layout: row 0 = north, col = east. Pixel (row, col) -> lat/lon
        max_lat = tile_lat + 1.0 - row / n
        max_lon = tile_lon + col / n

        total = int(steep_h + steep_v)
        return (basename, total, float(max_diff), max_lat, max_lon, None)
    except Exception as e:
        return (basename, 0, 0, 0, 0, str(e))


def scan_dat(args):
    filepath, threshold = args
    basename = os.path.basename(filepath)
    try:
        if filepath.endswith('.gz'):
            with gzip.open(filepath, 'rb') as f:
                data = f.read()
        else:
            with open(filepath, 'rb') as f:
                data = f.read()

        total_steep = 0
        max_diff = 0.0
        max_lat = 0.0
        max_lon = 0.0
        num_blocks = len(data) // IO_BLOCK_SIZE

        for i in range(num_blocks):
            block = data[i * IO_BLOCK_SIZE:(i + 1) * IO_BLOCK_SIZE]
            bitmap, blk_lat, blk_lon, crc, version, spacing = struct.unpack("<QiiHHH", block[:22])
            if version == 0 and spacing == 0:
                continue

            # Parse heights into numpy array (28 rows x 32 cols)
            offset = 22
            h = np.frombuffer(block[offset:offset + TERRAIN_GRID_BLOCK_SIZE_X * TERRAIN_GRID_BLOCK_SIZE_Y * 2],
                              dtype='<i2').reshape(TERRAIN_GRID_BLOCK_SIZE_X, TERRAIN_GRID_BLOCK_SIZE_Y).astype(np.float32)

            diff_h = np.abs(np.diff(h, axis=1))
            diff_v = np.abs(np.diff(h, axis=0))

            steep_h = np.sum(diff_h > threshold)
            steep_v = np.sum(diff_v > threshold)
            total_steep += int(steep_h + steep_v)

            blk_max_h = np.max(diff_h) if diff_h.size > 0 else 0
            blk_max_v = np.max(diff_v) if diff_v.size > 0 else 0
            blk_max = max(blk_max_h, blk_max_v)
            if blk_max > max_diff:
                max_diff = float(blk_max)
                # Block lat/lon is SW corner in 1e7, spacing in metres
                # gx = north index, gy = east index
                if blk_max_h >= blk_max_v:
                    idx = np.argmax(diff_h)
                    gx, gy = divmod(idx, diff_h.shape[1])
                else:
                    idx = np.argmax(diff_v)
                    gx, gy = divmod(idx, diff_v.shape[1])
                max_lat = blk_lat * 1e-7 + gx * spacing / 111111.0
                max_lon = blk_lon * 1e-7 + gy * spacing / (111111.0 * max(math.cos(math.radians(blk_lat * 1e-7)), 0.01))

        return (basename, total_steep, max_diff, max_lat, max_lon, None)
    except Exception as e:
        return (basename, 0, 0, 0, 0, str(e))


def main():
    parser = argparse.ArgumentParser(
        description='Find terrain files with steep height differences between neighbours')
    parser.add_argument('paths', nargs='+',
                        help='Directories and/or individual .hgt.zip / .DAT.gz files')
    parser.add_argument('--threshold', type=float, default=1000,
                        help='Height difference threshold in metres (default: 1000)')
    parser.add_argument('--processes', type=int, default=8,
                        help='Number of parallel workers')
    parser.add_argument('-o', type=str, default=None,
                        help='Save report to file')
    args = parser.parse_args()

    files = []
    for path in args.paths:
        if os.path.isdir(path):
            for f in glob.glob(os.path.join(path, '**/*.hgt.zip'), recursive=True):
                files.append(f)
            for f in glob.glob(os.path.join(path, '**/*.DAT.gz'), recursive=True):
                files.append(f)
            for f in glob.glob(os.path.join(path, '**/*.DAT'), recursive=True):
                if not f.endswith('.DAT.gz'):
                    files.append(f)
        elif os.path.isfile(path):
            files.append(path)
    files.sort()

    if not files:
        print("No terrain files found")
        sys.exit(1)

    hgt_work = [(f, args.threshold) for f in files if file_type(f) == 'hgt']
    dat_work = [(f, args.threshold) for f in files if file_type(f) == 'dat']

    print("Scanning %d HGT + %d DAT files, threshold=%gm, %d processes..." % (
        len(hgt_work), len(dat_work), args.threshold, args.processes))

    results = []
    with Pool(processes=args.processes) as pool:
        scanned = 0
        total = len(hgt_work) + len(dat_work)
        if hgt_work:
            for r in pool.imap_unordered(scan_hgt, hgt_work, chunksize=16):
                scanned += 1
                results.append(r)
                if scanned % 1000 == 0:
                    print("  %d / %d" % (scanned, total))
        if dat_work:
            for r in pool.imap_unordered(scan_dat, dat_work, chunksize=16):
                scanned += 1
                results.append(r)
                if scanned % 1000 == 0:
                    print("  %d / %d" % (scanned, total))

    results.sort(key=lambda x: x[0])

    steep = [(name, count, maxd, lat, lon) for name, count, maxd, lat, lon, err in results if count > 0 and err is None]
    errors = [(name, err) for name, _, _, _, _, err in results if err is not None]

    # Sort by max difference descending
    steep.sort(key=lambda x: -x[2])

    lines = []
    lines.append("=== Steep terrain report (threshold=%gm) ===" % args.threshold)
    lines.append("Files scanned: %d" % len(results))
    lines.append("Files exceeding threshold: %d" % len(steep))
    lines.append("")

    if steep:
        lines.append("%-30s %8s %8s  %s" % ("File", "Count", "Max(m)", "Max location"))
        lines.append("-" * 72)
        for name, count, maxd, lat, lon in steep:
            lines.append("%-30s %8d %8.0f  %.4f,%.4f" % (name, count, maxd, lat, lon))
        lines.append("")

    if errors:
        lines.append("Errors: %d" % len(errors))
        for name, err in errors:
            lines.append("  %s: %s" % (name, err))

    report = "\n".join(lines)
    print(report)

    if args.o:
        with open(args.o, 'w') as f:
            f.write(report + "\n")
        print("\nReport written to %s" % args.o)


if __name__ == '__main__':
    main()
