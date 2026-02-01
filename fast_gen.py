#!/usr/bin/env python3
"""
Fast HGT-to-DAT converter for ArduPilot terrain files.

Replaces the slow offline_gen.py pipeline with numpy-vectorised conversion.
Reads .hgt.zip files directly and produces .DAT.gz files.

Usage:
    python3 fast_gen.py <hgt_dir> <output_dir> [options]
"""

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
import fastcrc

from terrain_gen import (
    add_offset,
    east_blocks,
    get_distance_NE_e7,
    longitude_scale,
    pos_from_file_offset,
    to_float32,
    IO_BLOCK_SIZE,
    IO_BLOCK_DATA_SIZE,
    IO_BLOCK_TRAILER_SIZE,
    TERRAIN_GRID_BLOCK_SIZE_X,
    TERRAIN_GRID_BLOCK_SIZE_Y,
    TERRAIN_GRID_BLOCK_SPACING_X,
    TERRAIN_GRID_BLOCK_SPACING_Y,
    TERRAIN_GRID_FORMAT_VERSION,
    LOCATION_SCALING_FACTOR,
    LOCATION_SCALING_FACTOR_INV,
    GridBlock,
)

BITMAP = (1 << 56) - 1

# Filename pattern: N00E006.hgt.zip or S45W067.hgt.zip
HGT_FILENAME_RE = re.compile(r'([NS])(\d{2})([EW])(\d{3})\.hgt\.zip$')


def parse_hgt_filename(filepath):
    """Parse lat/lon from an HGT filename. Returns (lat, lon) integers."""
    basename = os.path.basename(filepath)
    m = HGT_FILENAME_RE.match(basename)
    if m is None:
        return None
    lat = int(m.group(2))
    lon = int(m.group(4))
    if m.group(1) == 'S':
        lat = -lat
    if m.group(3) == 'W':
        lon = -lon
    return (lat, lon)


def load_hgt_zip(filepath):
    """Load an HGT zip file and return elevation data as a numpy array.

    Returns array with shape (size, size), row 0 = south, row N = north.
    Values are int16. Void values (-32768) are replaced with 0.
    """
    with zipfile.ZipFile(filepath, 'r') as zf:
        names = zf.namelist()
        if len(names) != 1:
            raise ValueError(f"Expected 1 file in zip, got {len(names)}: {filepath}")
        data = zf.read(names[0])

    size = int(math.sqrt(len(data) // 2))
    if size not in (1201, 3601):
        raise ValueError(f"Unexpected HGT size {size} in {filepath}")

    # HGT files are big-endian int16, row 0 = north
    arr = np.frombuffer(data, dtype='>i2').reshape(size, size).astype(np.int16)
    # Flip so row 0 = south
    arr = arr[::-1].copy()
    # Replace void values
    arr[arr == -32768] = 0
    return arr


def build_combined_array(lat_int, lon_int, hgt_map, hgt_cache):
    """Build a combined elevation array covering the main tile and neighbours.

    Returns (combined_array, min_lat, min_lon, hgt_size).
    The combined array covers a 2x2 degree area (or less if neighbours missing).
    Row 0 = south, col 0 = west.
    """
    # We need the main tile plus potentially tiles to the north and east
    # (for edge blocks that interpolate across the degree boundary)
    needed = [
        (lat_int, lon_int),
        (lat_int, lon_int + 1),
        (lat_int + 1, lon_int),
        (lat_int + 1, lon_int + 1),
    ]

    # Load needed tiles
    hgt_size = None
    for (lat, lon) in needed:
        if (lat, lon) in hgt_cache:
            if hgt_cache[(lat, lon)] is not None:
                hgt_size = hgt_cache[(lat, lon)].shape[0]
            continue
        if (lat, lon) in hgt_map:
            try:
                arr = load_hgt_zip(hgt_map[(lat, lon)])
                hgt_cache[(lat, lon)] = arr
                hgt_size = arr.shape[0]
            except Exception as e:
                print(f"Warning: failed to load {hgt_map[(lat, lon)]}: {e}")
                hgt_cache[(lat, lon)] = None
        else:
            hgt_cache[(lat, lon)] = None

    if hgt_size is None:
        # No tiles at all - pure ocean
        hgt_size = 3601  # default to SRTM1 size
        # Return zeros
        combined = np.zeros((2 * (hgt_size - 1) + 1, 2 * (hgt_size - 1) + 1), dtype=np.int16)
        return combined, lat_int, lon_int, hgt_size

    # Build combined array for 2x2 degree area
    n = hgt_size - 1  # pixels per degree
    combined = np.zeros((2 * n + 1, 2 * n + 1), dtype=np.int16)

    for (lat, lon) in needed:
        tile = hgt_cache.get((lat, lon))
        if tile is None:
            continue
        row_offset = (lat - lat_int) * n
        col_offset = (lon - lon_int) * n
        combined[row_offset:row_offset + hgt_size, col_offset:col_offset + hgt_size] = tile

    return combined, lat_int, lon_int, hgt_size


def enumerate_valid_blocks(lat_int, lon_int, spacing, fmt):
    """Enumerate all valid blocks for a 1-degree tile.

    Returns list of (blocknum, grid_idx_x, grid_idx_y, lat_e7, lon_e7).
    """
    ref_lat = lat_int * 10 * 1000 * 1000
    ref_lon = lon_int * 10 * 1000 * 1000
    stride = east_blocks(ref_lat, ref_lon, spacing, fmt)

    valid_blocks = []
    blocknum = -1

    while True:
        blocknum += 1
        (lat_e7, lon_e7) = pos_from_file_offset(lat_int, lon_int,
                                                  blocknum * IO_BLOCK_SIZE,
                                                  spacing, fmt)
        if lat_e7 * 1.0e-7 - lat_int >= 1.0:
            break

        # Validate round-trip: compute grid indices and check blocknum matches
        grid_idx_x = blocknum // stride
        grid_idx_y = blocknum % stride

        # Recompute position from grid indices (same as GridBlock.__init__)
        idx_x = grid_idx_x * TERRAIN_GRID_BLOCK_SPACING_X
        idx_y = grid_idx_y * TERRAIN_GRID_BLOCK_SPACING_Y

        (check_lat, check_lon) = add_offset(
            ref_lat, ref_lon,
            idx_x * float(spacing),
            idx_y * float(spacing),
            fmt
        )

        # Compute offset back and verify block number
        offset = get_distance_NE_e7(ref_lat, ref_lon, check_lat, check_lon, fmt)
        check_idx_x = int(round(offset[0]) / spacing) // TERRAIN_GRID_BLOCK_SPACING_X
        check_idx_y = int(round(offset[1]) / spacing) // TERRAIN_GRID_BLOCK_SPACING_Y
        check_blocknum = stride * check_idx_x + check_idx_y

        if check_blocknum != blocknum:
            continue

        valid_blocks.append((blocknum, grid_idx_x, grid_idx_y, check_lat, check_lon))

    return valid_blocks, stride


def compute_grid_points_vectorised(valid_blocks, spacing, fmt):
    """Compute lat/lon coordinates for all grid points in all blocks.

    Returns (point_lat_e7, point_lon_e7) each with shape (n_blocks, 28, 32).
    The grid is 28 points north (gx) by 32 points east (gy), matching
    TERRAIN_GRID_BLOCK_SIZE_X=28, TERRAIN_GRID_BLOCK_SIZE_Y=32.

    The original create_degree() converts block SW corners through a float
    round-trip (int -> *1e-7 -> *1e7) before calling add_offset. This
    introduces tiny float64 rounding errors that we must replicate exactly
    for bit-identical output.
    """
    n_blocks = len(valid_blocks)

    # Block SW corners as integers
    sw_lat_int = np.array([b[3] for b in valid_blocks], dtype=np.int64)
    sw_lon_int = np.array([b[4] for b in valid_blocks], dtype=np.int64)

    # Apply the same float round-trip as create_degree():
    #   lat = lat_e7 * 1.0e-7;  add_offset(lat*1.0e7, ...)
    sw_lat_f = sw_lat_int.astype(np.float64) * 1e-7 * 1e7  # (n_blocks,)
    sw_lon_f = sw_lon_int.astype(np.float64) * 1e-7 * 1e7  # (n_blocks,)

    LSCF_INV = float(LOCATION_SCALING_FACTOR_INV)

    # North offsets for gx: 0..27
    gx_range = np.arange(TERRAIN_GRID_BLOCK_SIZE_X, dtype=np.float64)  # 28
    north_m = gx_range * spacing

    # dlat = int(float(ofs_north) * LSCF_INV) — same for all blocks, shape (28,)
    dlat = np.trunc(north_m * LSCF_INV).astype(np.int64)

    # Point latitudes = int(sw_lat_f + dlat), shape (n_blocks, 28)
    point_lat_e7 = np.trunc(sw_lat_f[:, None] + dlat[None, :].astype(np.float64)).astype(np.int64)

    # East offsets for gy: 0..31
    gy_range = np.arange(TERRAIN_GRID_BLOCK_SIZE_Y, dtype=np.float64)  # 32
    east_m = gy_range * spacing

    # 4.1 format: longitude_scale uses midpoint (lat_e7 + dlat*0.5) * 1e-7
    # where lat_e7 is sw_lat_f (the float round-tripped value)
    mid_lat_deg = (sw_lat_f[:, None] + dlat[None, :].astype(np.float64) * 0.5) * 1e-7
    # (n_blocks, 28)

    # Match float32 behaviour for longitude_scale
    rad_f32 = np.radians(mid_lat_deg).astype(np.float32)
    lon_scale = np.cos(rad_f32.astype(np.float64)).astype(np.float32).astype(np.float64)
    lon_scale = np.maximum(lon_scale, 0.01)  # (n_blocks, 28)

    # dlng = int(float(ofs_east) * LSCF_INV / lon_scale), shape (n_blocks, 28, 32)
    dlng = np.trunc(east_m[None, None, :] * LSCF_INV / lon_scale[:, :, None]).astype(np.int64)

    # Point longitudes = int(sw_lon_f + dlng), shape (n_blocks, 28, 32)
    point_lon_e7 = np.trunc(sw_lon_f[:, None, None] + dlng.astype(np.float64)).astype(np.int64)

    # Expand lat to (n_blocks, 28, 32)
    point_lat_e7 = np.broadcast_to(point_lat_e7[:, :, None],
                                    (n_blocks, TERRAIN_GRID_BLOCK_SIZE_X,
                                     TERRAIN_GRID_BLOCK_SIZE_Y)).copy()

    return point_lat_e7, point_lon_e7


def interpolate_heights(point_lat_e7, point_lon_e7, combined, min_lat, min_lon, hgt_size):
    """Bilinear interpolation of heights from the combined HGT array.

    point_lat_e7, point_lon_e7: shape (n_blocks, 28, 32), int64
    combined: the combined elevation array, row 0 = south at min_lat
    Returns heights array shape (n_blocks, 28, 32), int16.
    """
    n = hgt_size - 1  # pixels per degree

    # Convert to float pixel coordinates in the combined array
    cy = (point_lat_e7.astype(np.float64) * 1e-7 - min_lat) * n
    cx = (point_lon_e7.astype(np.float64) * 1e-7 - min_lon) * n

    cy_int = np.floor(cy).astype(np.int32)
    cx_int = np.floor(cx).astype(np.int32)
    cy_frac = cy - cy_int
    cx_frac = cx - cx_int

    # Clamp to valid range
    max_row = combined.shape[0] - 2
    max_col = combined.shape[1] - 2
    cy_int = np.clip(cy_int, 0, max_row)
    cx_int = np.clip(cx_int, 0, max_col)

    # Flatten for advanced indexing
    flat_cy = cy_int.ravel()
    flat_cx = cx_int.ravel()
    shape = cy_int.shape

    v00 = combined[flat_cy, flat_cx].reshape(shape).astype(np.float64)
    v10 = combined[flat_cy, flat_cx + 1].reshape(shape).astype(np.float64)
    v01 = combined[flat_cy + 1, flat_cx].reshape(shape).astype(np.float64)
    v11 = combined[flat_cy + 1, flat_cx + 1].reshape(shape).astype(np.float64)

    # Handle void values: the original code treats -1 (from getPixelValue) as None
    # and uses _avg which skips None values. Since we replaced -32768 with 0 on load,
    # this matches the original behaviour (void -> 0 altitude).

    # Bilinear interpolation — two-step to match original scalar code exactly.
    # The original does: interp x first, then interp y on the results.
    # A single 4-point formula has different rounding and can be off-by-one.
    val_x0 = v10 * cx_frac + v00 * (1 - cx_frac)  # y_int row
    val_x1 = v11 * cx_frac + v01 * (1 - cx_frac)  # y_int+1 row
    heights = val_x1 * cy_frac + val_x0 * (1 - cy_frac)

    return heights.astype(np.int16)


def pack_dat_file(valid_blocks, heights, lat_int, lon_int, spacing, fmt):
    """Pack all blocks into a DAT file buffer.

    valid_blocks: list of (blocknum, grid_idx_x, grid_idx_y, lat_e7, lon_e7)
    heights: shape (n_blocks, 28, 32) int16
    Returns bytes.
    """
    if not valid_blocks:
        return b''

    max_blocknum = max(b[0] for b in valid_blocks)
    file_buf = bytearray((max_blocknum + 1) * IO_BLOCK_SIZE)

    for i, (blocknum, grid_idx_x, grid_idx_y, blk_lat, blk_lon) in enumerate(valid_blocks):
        # Build block with crc=0 first
        header = struct.pack('<QiiHHH', BITMAP, int(blk_lat), int(blk_lon),
                             0, TERRAIN_GRID_FORMAT_VERSION, spacing)

        # Heights for this block: shape (28, 32), pack as little-endian int16
        # The original packs gx rows sequentially, each row being 32 int16 values
        h_block = heights[i]  # (28, 32)
        h_bytes = h_block.astype('<i2').tobytes()

        # Trailer: grid_idx_x (H), grid_idx_y (H), lon_degrees (h), lat_degrees (b)
        trailer = struct.pack('<HHhb', grid_idx_x, grid_idx_y, lon_int, lat_int)

        # Assemble block: header(22) + heights(1792) + trailer(7) = 1821 data bytes
        # Plus padding to fill IO_BLOCK_SIZE
        block = header + h_bytes + trailer
        padding_size = IO_BLOCK_SIZE - len(block)
        block = block + b'\x00' * padding_size

        # Compute CRC over first 1821 bytes (with crc field = 0)
        crc = fastcrc.crc16.xmodem(block[:IO_BLOCK_DATA_SIZE])

        # Patch CRC into the block (at offset 16, 2 bytes)
        block = block[:16] + struct.pack('<H', crc) + block[18:]

        file_buf[blocknum * IO_BLOCK_SIZE:(blocknum + 1) * IO_BLOCK_SIZE] = block

    return bytes(file_buf)


def dat_filename(lat_int, lon_int):
    """Generate the DAT filename for a lat/lon pair."""
    ns = 'S' if lat_int < 0 else 'N'
    ew = 'W' if lon_int < 0 else 'E'
    return "%c%02u%c%03u.DAT.gz" % (ns, min(abs(lat_int), 99),
                                     ew, min(abs(lon_int), 999))


def process_tile(hgt_file, hgt_map, output_dir, spacing, fmt):
    """Process a single HGT tile to produce a DAT.gz file."""
    coords = parse_hgt_filename(hgt_file)
    if coords is None:
        print(f"Skipping unrecognised file: {hgt_file}")
        return

    lat_int, lon_int = coords
    outname = dat_filename(lat_int, lon_int)
    outpath = os.path.join(output_dir, outname)

    if os.path.exists(outpath):
        return

    hgt_cache = {}

    # Step 1: Enumerate valid blocks
    valid_blocks, stride = enumerate_valid_blocks(lat_int, lon_int, spacing, fmt)
    if not valid_blocks:
        print(f"No valid blocks for {os.path.basename(hgt_file)}")
        return

    # Step 2: Build combined elevation array
    combined, min_lat, min_lon, hgt_size = build_combined_array(
        lat_int, lon_int, hgt_map, hgt_cache)

    # Step 3: Compute grid point coordinates (vectorised, in chunks to limit memory)
    CHUNK_SIZE = 2000
    n_blocks = len(valid_blocks)
    all_heights = np.zeros((n_blocks, TERRAIN_GRID_BLOCK_SIZE_X,
                            TERRAIN_GRID_BLOCK_SIZE_Y), dtype=np.int16)

    for chunk_start in range(0, n_blocks, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, n_blocks)
        chunk_blocks = valid_blocks[chunk_start:chunk_end]

        point_lat_e7, point_lon_e7 = compute_grid_points_vectorised(
            chunk_blocks, spacing, fmt)

        # Step 4: Interpolate heights
        chunk_heights = interpolate_heights(
            point_lat_e7, point_lon_e7, combined, min_lat, min_lon, hgt_size)

        all_heights[chunk_start:chunk_end] = chunk_heights

    # Step 5: Pack into DAT file buffer
    file_buf = pack_dat_file(valid_blocks, all_heights, lat_int, lon_int, spacing, fmt)

    # Step 6: Compress and write atomically
    os.makedirs(output_dir, exist_ok=True)
    tmp_path = outpath + '.tmp'
    with gzip.open(tmp_path, 'wb') as f:
        f.write(file_buf)
    os.rename(tmp_path, outpath)

    print(f"Generated {outname} ({n_blocks} blocks)")


def process_tile_wrapper(args):
    """Wrapper for multiprocessing that unpacks arguments."""
    try:
        process_tile(*args)
    except Exception as e:
        print(f"Error processing {args[0]}: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description='Fast HGT-to-DAT converter for ArduPilot terrain files')
    parser.add_argument('hgt_dir', help='Directory containing .hgt.zip files')
    parser.add_argument('output_dir', help='Output directory for .DAT.gz files')
    parser.add_argument('--spacing', type=int, default=30, choices=[30, 100],
                        help='Grid spacing in metres (default: 30)')
    parser.add_argument('--processes', type=int, default=8,
                        help='Number of parallel workers (default: 8)')
    args = parser.parse_args()

    # Scan for HGT files (flat or continent subdirs)
    hgt_files = sorted(glob.glob(os.path.join(args.hgt_dir, '**/*.hgt.zip'),
                                 recursive=True))
    if not hgt_files:
        print(f"No .hgt.zip files found in {args.hgt_dir}")
        sys.exit(1)

    print(f"Found {len(hgt_files)} HGT files")

    # Build lookup dict: (lat, lon) -> filepath
    hgt_map = {}
    for f in hgt_files:
        coords = parse_hgt_filename(f)
        if coords is not None:
            hgt_map[coords] = f

    os.makedirs(args.output_dir, exist_ok=True)

    # Count how many already exist
    existing = sum(1 for f in hgt_files
                   if os.path.exists(os.path.join(args.output_dir,
                                                  dat_filename(*parse_hgt_filename(f)))))
    if existing > 0:
        print(f"Skipping {existing} already-generated tiles")

    work_args = [(f, hgt_map, args.output_dir, args.spacing, "4.1")
                 for f in hgt_files]

    if args.processes <= 1:
        for wa in work_args:
            process_tile_wrapper(wa)
    else:
        with Pool(processes=args.processes) as pool:
            pool.map(process_tile_wrapper, work_args)

    print("Done!")


if __name__ == '__main__':
    main()
