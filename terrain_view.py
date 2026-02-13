#!/usr/bin/env python3
"""
Visualise ArduPilot terrain .DAT files or SRTM .hgt files as 2D images.

Height 0 is black. Non-zero heights are coloured blue (lowest) to red (highest).
Mouse cursor position shows the lat/lon and height.

Usage:
    python3 terrain_view.py <file.DAT or file.DAT.gz or file.hgt.zip>
    python3 terrain_view.py --diff <file1> <file2>
"""

import argparse
import gzip
import math
import os
import re
import struct
import sys
import zipfile
import numpy as np

IO_BLOCK_SIZE = 2048
GRID_SIZE_X = 28   # latitude (north), 7 * 4
GRID_SIZE_Y = 32   # longitude (east), 8 * 4
BLOCK_SPACING_X = 24  # (7-1) * 4, blocks overlap by 4
BLOCK_SPACING_Y = 28  # (8-1) * 4, blocks overlap by 4

# HGT filename pattern
HGT_FILENAME_RE = re.compile(r'([NS])(\d{2})([EW])(\d{3})\.hgt(?:\.zip)?$', re.IGNORECASE)


def parse_hgt_filename(filepath):
    """Parse lat/lon from an HGT filename. Returns (lat, lon) or None."""
    basename = os.path.basename(filepath)
    m = HGT_FILENAME_RE.match(basename)
    if m is None:
        return None
    lat = int(m.group(2))
    lon = int(m.group(4))
    if m.group(1).upper() == 'S':
        lat = -lat
    if m.group(3).upper() == 'W':
        lon = -lon
    return (lat, lon)


def read_hgt_file(filepath):
    """Read an HGT or HGT.zip file and return (height_grid, tile_lat, tile_lon, spacing).

    Returns grid with row 0 = south (same orientation as DAT files).
    """
    coords = parse_hgt_filename(filepath)
    if coords is None:
        raise ValueError(f"Cannot parse lat/lon from filename: {filepath}")

    tile_lat, tile_lon = coords

    if filepath.lower().endswith('.zip'):
        with zipfile.ZipFile(filepath, 'r') as zf:
            names = zf.namelist()
            if len(names) != 1:
                raise ValueError(f"Expected 1 file in zip, got {len(names)}: {filepath}")
            data = zf.read(names[0])
    else:
        with open(filepath, 'rb') as f:
            data = f.read()

    size = int(math.sqrt(len(data) // 2))
    if size not in (1201, 3601):
        raise ValueError(f"Unexpected HGT size {size} in {filepath}")

    # HGT files are big-endian int16, row 0 = north
    arr = np.frombuffer(data, dtype='>i2').reshape(size, size).astype(np.int16)
    # Flip so row 0 = south (consistent with DAT grid orientation)
    arr = arr[::-1].copy()

    # Spacing: SRTM3 (1201) = ~90m, SRTM1 (3601) = ~30m
    # More precisely: 1 degree / (size-1) pixels, converted to metres at equator
    spacing = int(round(111320.0 / (size - 1)))

    return arr, tile_lat, tile_lon, spacing


def is_hgt_file(filepath):
    """Check if filepath is an HGT file."""
    lower = filepath.lower()
    return lower.endswith('.hgt') or lower.endswith('.hgt.zip')


def read_dat_file(filepath):
    """Read all blocks from a DAT or DAT.gz file."""
    if filepath.endswith('.gz'):
        with gzip.open(filepath, 'rb') as f:
            data = f.read()
    else:
        with open(filepath, 'rb') as f:
            data = f.read()

    blocks = []
    num_blocks = len(data) // IO_BLOCK_SIZE
    tile_lat = None
    tile_lon = None
    spacing = None

    for i in range(num_blocks):
        block_data = data[i * IO_BLOCK_SIZE:(i + 1) * IO_BLOCK_SIZE]
        if len(block_data) < IO_BLOCK_SIZE:
            break

        bitmap, lat_e7, lon_e7, crc, version, blk_spacing = struct.unpack('<QiiHHH', block_data[:22])

        if bitmap == 0 and lat_e7 == 0 and lon_e7 == 0:
            continue

        height_data = block_data[22:22 + GRID_SIZE_X * GRID_SIZE_Y * 2]
        if len(height_data) < GRID_SIZE_X * GRID_SIZE_Y * 2:
            continue

        heights = np.zeros((GRID_SIZE_X, GRID_SIZE_Y), dtype=np.int16)
        for gx in range(GRID_SIZE_X):
            offset = gx * GRID_SIZE_Y * 2
            heights[gx, :] = struct.unpack('<%dh' % GRID_SIZE_Y,
                                           height_data[offset:offset + GRID_SIZE_Y * 2])

        trailer_offset = 22 + GRID_SIZE_X * GRID_SIZE_Y * 2
        grid_idx_x, grid_idx_y, lon_deg, lat_deg = struct.unpack('<HHhb', block_data[trailer_offset:trailer_offset + 7])

        if tile_lat is None:
            tile_lat = lat_deg
            tile_lon = lon_deg
            spacing = blk_spacing

        blocks.append({
            'grid_idx_x': grid_idx_x,
            'grid_idx_y': grid_idx_y,
            'heights': heights,
        })

    return blocks, tile_lat, tile_lon, spacing


def blocks_to_grid(blocks):
    """Assemble blocks into a single height grid using block indices.

    Blocks overlap by 4 grid points in each direction, so placement
    uses BLOCK_SPACING rather than GRID_SIZE.
    """
    max_idx_x = max(b['grid_idx_x'] for b in blocks)
    max_idx_y = max(b['grid_idx_y'] for b in blocks)

    total_x = max_idx_x * BLOCK_SPACING_X + GRID_SIZE_X
    total_y = max_idx_y * BLOCK_SPACING_Y + GRID_SIZE_Y

    height_grid = np.zeros((total_x, total_y), dtype=np.int16)

    for block in blocks:
        x_start = block['grid_idx_x'] * BLOCK_SPACING_X
        y_start = block['grid_idx_y'] * BLOCK_SPACING_Y
        height_grid[x_start:x_start + GRID_SIZE_X,
                    y_start:y_start + GRID_SIZE_Y] = block['heights']

    return height_grid


def height_to_rgb(height_grid, grey=False):
    """Convert height grid to RGB image.

    0 = black. Non-zero heights mapped blue-to-red (default) or greyscale.
    """
    rgb = np.zeros((*height_grid.shape, 3), dtype=np.uint8)

    nonzero = height_grid != 0
    if not np.any(nonzero):
        return rgb

    nz_min = height_grid[nonzero].min()
    nz_max = height_grid[nonzero].max()

    if nz_max == nz_min:
        rgb[nonzero] = [128, 128, 128] if grey else [128, 0, 128]
        return rgb

    # Normalise non-zero heights to 0..1
    t = (height_grid[nonzero].astype(np.float32) - nz_min) / (nz_max - nz_min)

    if grey:
        v = (t * 255).astype(np.uint8)
        rgb[nonzero, 0] = v
        rgb[nonzero, 1] = v
        rgb[nonzero, 2] = v
    else:
        # Blue (0,0,255) at t=0 to Red (255,0,0) at t=1
        rgb[nonzero, 0] = (t * 255).astype(np.uint8)         # R
        rgb[nonzero, 1] = 0                                   # G
        rgb[nonzero, 2] = ((1 - t) * 255).astype(np.uint8)   # B

    return rgb


def load_file(filepath):
    """Load a DAT or HGT file and return the height grid and tile info, or None."""
    print(f"Reading {filepath}...")

    if is_hgt_file(filepath):
        try:
            height_grid, tile_lat, tile_lon, spacing = read_hgt_file(filepath)
            print(f"  HGT grid: {height_grid.shape[0]} x {height_grid.shape[1]}")
        except Exception as e:
            print(f"  Error reading HGT file: {e}")
            return None, None, None, None
    else:
        blocks, tile_lat, tile_lon, spacing = read_dat_file(filepath)
        print(f"  {len(blocks)} blocks")

        if not blocks:
            print(f"  No valid blocks found in {filepath}")
            return None, None, None, None

        height_grid = blocks_to_grid(blocks)
        print(f"  Grid: {height_grid.shape[0]} x {height_grid.shape[1]}")

    print(f"  Tile: lat={tile_lat}, lon={tile_lon}, spacing={spacing}m")

    nz = height_grid[height_grid != 0]
    if len(nz):
        print(f"  Altitude: {nz.min()}m to {nz.max()}m")

    return height_grid, tile_lat, tile_lon, spacing


def grid_to_latlon(grid_x, grid_y, tile_lat, tile_lon, spacing):
    """Convert grid coordinates to lat/lon.

    grid_x is north (latitude), grid_y is east (longitude).
    spacing is in metres.
    """
    # Approximate metres per degree
    lat_m_per_deg = 111320.0
    lon_m_per_deg = 111320.0 * np.cos(np.radians(tile_lat + 0.5))

    lat = tile_lat + (grid_x * spacing) / lat_m_per_deg
    lon = tile_lon + (grid_y * spacing) / lon_m_per_deg
    return lat, lon


def show_file(datfile, plt, grey=False):
    """Open a viewer window for one DAT file."""
    height_grid, tile_lat, tile_lon, spacing = load_file(datfile)
    if height_grid is None:
        return

    rgb = height_to_rgb(height_grid, grey=grey)

    # Flip so north is up (row 0 = bottom of grid = south)
    rgb = rgb[::-1, :, :]
    height_display = height_grid[::-1, :]
    grid_rows = height_display.shape[0]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(rgb, aspect='equal', interpolation='nearest')
    ax.set_axis_off()
    fig.suptitle(datfile)
    fig.canvas.manager.set_window_title(datfile)

    def format_coord(x, y):
        ix, iy = int(x + 0.5), int(y + 0.5)
        if 0 <= iy < height_display.shape[0] and 0 <= ix < height_display.shape[1]:
            h = height_display[iy, ix]
            # Convert display coords back to grid coords (flip y)
            grid_x = grid_rows - 1 - iy
            grid_y = ix
            lat, lon = grid_to_latlon(grid_x, grid_y, tile_lat, tile_lon, spacing)
            return f'lat={lat:.6f}  lon={lon:.6f}  height={h}m'
        return ''

    ax.format_coord = format_coord
    plt.tight_layout()


def diff_to_rgb(diff_grid):
    """Convert a signed difference grid to RGB.

    0 = black. Negative = blue, positive = red.
    Uses 98th percentile for scaling so outliers don't wash out small diffs.
    """
    rgb = np.zeros((*diff_grid.shape, 3), dtype=np.uint8)

    nonzero = diff_grid != 0
    if not np.any(nonzero):
        return rgb

    # Use 98th percentile instead of max to avoid outliers washing out the image
    abs_vals = np.abs(diff_grid[nonzero])
    scale = np.percentile(abs_vals, 98)
    if scale == 0:
        scale = abs_vals.max()
    if scale == 0:
        return rgb

    # Normalise to -1..1, clipping outliers
    t = np.clip(diff_grid[nonzero].astype(np.float32) / scale, -1, 1)

    # Positive (red), negative (blue)
    pos = t > 0
    neg = t < 0

    nz_indices = np.where(nonzero)
    pos_idx = tuple(idx[pos] for idx in nz_indices)
    neg_idx = tuple(idx[neg] for idx in nz_indices)

    rgb[pos_idx[0], pos_idx[1], 0] = (t[pos] * 255).astype(np.uint8)
    rgb[neg_idx[0], neg_idx[1], 2] = (-t[neg] * 255).astype(np.uint8)

    return rgb


def resample_grid(grid, old_spacing, new_spacing):
    """Resample a grid to a different spacing using nearest-neighbour."""
    if old_spacing == new_spacing:
        return grid
    scale = old_spacing / new_spacing
    new_rows = int(grid.shape[0] * scale)
    new_cols = int(grid.shape[1] * scale)
    # Create index arrays for nearest-neighbour resampling
    row_idx = (np.arange(new_rows) / scale).astype(int)
    col_idx = (np.arange(new_cols) / scale).astype(int)
    row_idx = np.clip(row_idx, 0, grid.shape[0] - 1)
    col_idx = np.clip(col_idx, 0, grid.shape[1] - 1)
    return grid[row_idx][:, col_idx]


def show_diff(file1, file2, plt, grey=False):
    """Show two files and their difference."""
    grid1, tile_lat1, tile_lon1, spacing1 = load_file(file1)
    grid2, tile_lat2, tile_lon2, spacing2 = load_file(file2)

    if grid1 is None or grid2 is None:
        return

    # Use first file's tile info for coordinate display
    tile_lat = tile_lat1 if tile_lat1 is not None else tile_lat2
    tile_lon = tile_lon1 if tile_lon1 is not None else tile_lon2

    # Resample to finer spacing if different
    if spacing1 != spacing2:
        target_spacing = min(spacing1, spacing2)
        print(f"  Resampling to {target_spacing}m spacing for comparison")
        grid1 = resample_grid(grid1, spacing1, target_spacing)
        grid2 = resample_grid(grid2, spacing2, target_spacing)
        spacing = target_spacing
    else:
        spacing = spacing1

    # Pad smaller grid to match larger
    rows = max(grid1.shape[0], grid2.shape[0])
    cols = max(grid1.shape[1], grid2.shape[1])

    def pad_grid(g):
        if g.shape[0] < rows or g.shape[1] < cols:
            p = np.zeros((rows, cols), dtype=g.dtype)
            p[:g.shape[0], :g.shape[1]] = g
            return p
        return g

    grid1 = pad_grid(grid1)
    grid2 = pad_grid(grid2)

    diff = grid1.astype(np.int32) - grid2.astype(np.int32)
    n_diff = np.count_nonzero(diff)
    print(f"  Diff: {n_diff} pixels differ, range {diff.min()} to {diff.max()}")

    rgb1 = height_to_rgb(grid1, grey=grey)[::-1, :, :]
    rgb2 = height_to_rgb(grid2, grey=grey)[::-1, :, :]
    rgb_diff = diff_to_rgb(diff)[::-1, :, :]
    h1 = grid1[::-1, :]
    h2 = grid2[::-1, :]
    d = diff[::-1, :]
    grid_rows = h1.shape[0]
    disp_spacing = spacing  # capture for closure

    def format1(x, y):
        ix, iy = int(x + 0.5), int(y + 0.5)
        if 0 <= iy < h1.shape[0] and 0 <= ix < h1.shape[1]:
            grid_x = grid_rows - 1 - iy
            grid_y = ix
            lat, lon = grid_to_latlon(grid_x, grid_y, tile_lat, tile_lon, disp_spacing)
            return f'lat={lat:.6f}  lon={lon:.6f}  h1={h1[iy,ix]}m  h2={h2[iy,ix]}m  diff={d[iy,ix]}m'
        return ''

    # File 1
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    ax1.imshow(rgb1, aspect='equal', interpolation='nearest')
    ax1.set_axis_off()
    fig1.suptitle(file1)
    fig1.canvas.manager.set_window_title(file1)
    ax1.format_coord = format1
    plt.tight_layout()

    # File 2
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    ax2.imshow(rgb2, aspect='equal', interpolation='nearest')
    ax2.set_axis_off()
    fig2.suptitle(file2)
    fig2.canvas.manager.set_window_title(file2)
    ax2.format_coord = format1
    plt.tight_layout()

    # Diff
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    ax3.imshow(rgb_diff, aspect='equal', interpolation='nearest')
    ax3.set_axis_off()
    fig3.suptitle(f"Diff: {file1} - {file2}")
    fig3.canvas.manager.set_window_title(f"Diff ({n_diff} pixels differ)")
    ax3.format_coord = format1
    plt.tight_layout()


def main():
    parser = argparse.ArgumentParser(description='Visualise ArduPilot terrain DAT files or SRTM HGT files')
    parser.add_argument('datfiles', nargs='+', help='Path to .DAT, .DAT.gz, .hgt, or .hgt.zip files')
    parser.add_argument('--diff', action='store_true',
                        help='Show difference between exactly 2 files')
    parser.add_argument('--grey', action='store_true',
                        help='Show heights as greyscale instead of blue-to-red')
    args = parser.parse_args()

    if args.diff:
        if len(args.datfiles) != 2:
            print("--diff requires exactly 2 files")
            sys.exit(1)

    import matplotlib.pyplot as plt

    if args.diff:
        show_diff(args.datfiles[0], args.datfiles[1], plt, grey=args.grey)
    else:
        for datfile in args.datfiles:
            show_file(datfile, plt, grey=args.grey)

    plt.show()


if __name__ == '__main__':
    main()
