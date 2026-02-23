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
    # Replace void values with -1 to match srtm.py getPixelValue() behaviour
    arr[arr == -32768] = -1
    return arr


def load_tile_dict(lat_int, lon_int, hgt_map, hgt_cache):
    """Load the main tile and its neighbours into a dict.

    Returns (tile_dict, hgt_size) where tile_dict maps (lat, lon) to a
    numpy array (row 0 = south), or None for missing/ocean tiles.
    """
    needed = [
        (lat_int, lon_int),
        (lat_int, lon_int + 1),
        (lat_int + 1, lon_int),
        (lat_int + 1, lon_int + 1),
    ]

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
        hgt_size = 3601

    return hgt_cache, hgt_size


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


def compute_grid_points_vectorised(valid_blocks, lat_int, lon_int, spacing, fmt):
    """Compute lat/lon coordinates for all grid points in all blocks.

    Returns (point_lat_e7, point_lon_e7) each with shape (n_blocks, 28, 32).
    The grid is 28 points north (gx) by 32 points east (gy), matching
    TERRAIN_GRID_BLOCK_SIZE_X=28, TERRAIN_GRID_BLOCK_SIZE_Y=32.

    Uses a single-step offset from the degree corner for each grid point,
    matching what AP_Terrain does on the vehicle. This avoids the two-step
    error (degree corner -> block corner -> grid point) where compounding
    longitude_scale at different latitudes causes horizontal drift at high
    latitudes.
    """
    n_blocks = len(valid_blocks)
    ref_lat = lat_int * 10 * 1000 * 1000  # degree corner in e7
    ref_lon = lon_int * 10 * 1000 * 1000

    # Grid indices for each block
    block_grid_idx_x = np.array([b[1] for b in valid_blocks], dtype=np.int64)  # (n_blocks,)
    block_grid_idx_y = np.array([b[2] for b in valid_blocks], dtype=np.int64)  # (n_blocks,)

    LSCF_INV = float(LOCATION_SCALING_FACTOR_INV)

    # Global grid indices (single-step from degree corner)
    gx_range = np.arange(TERRAIN_GRID_BLOCK_SIZE_X, dtype=np.int64)  # 0..27
    gy_range = np.arange(TERRAIN_GRID_BLOCK_SIZE_Y, dtype=np.int64)  # 0..31

    # North: global_idx_x = grid_idx_x * SPACING_X + gx, shape (n_blocks, 28)
    global_idx_x = block_grid_idx_x[:, None] * TERRAIN_GRID_BLOCK_SPACING_X + gx_range[None, :]
    north_m = global_idx_x.astype(np.float64) * spacing

    # dlat = int(north_m * LSCF_INV), shape (n_blocks, 28)
    dlat = np.trunc(north_m * LSCF_INV).astype(np.int64)

    # Point latitudes: single-step from degree corner
    point_lat_e7 = ref_lat + dlat  # (n_blocks, 28)

    # East: global_idx_y = grid_idx_y * SPACING_Y + gy, shape (n_blocks, 32)
    global_idx_y = block_grid_idx_y[:, None] * TERRAIN_GRID_BLOCK_SPACING_Y + gy_range[None, :]
    east_m = global_idx_y.astype(np.float64) * spacing

    # 4.1 format: longitude_scale uses midpoint (ref_lat + dlat*0.5) * 1e-7
    mid_lat_deg = (float(ref_lat) + dlat.astype(np.float64) * 0.5) * 1e-7
    # (n_blocks, 28)

    # Match float32 behaviour for longitude_scale
    rad_f32 = np.radians(mid_lat_deg).astype(np.float32)
    lon_scale = np.cos(rad_f32.astype(np.float64)).astype(np.float32).astype(np.float64)
    lon_scale = np.maximum(lon_scale, 0.01)  # (n_blocks, 28)

    # dlng = int(east_m * LSCF_INV / lon_scale), shape (n_blocks, 28, 32)
    dlng = np.trunc(east_m[:, None, :] * LSCF_INV / lon_scale[:, :, None]).astype(np.int64)

    # Point longitudes: single-step from degree corner
    point_lon_e7 = ref_lon + dlng  # (n_blocks, 28, 32)

    # Expand lat to (n_blocks, 28, 32)
    point_lat_e7 = np.broadcast_to(point_lat_e7[:, :, None],
                                    (n_blocks, TERRAIN_GRID_BLOCK_SIZE_X,
                                     TERRAIN_GRID_BLOCK_SIZE_Y)).copy()

    return point_lat_e7, point_lon_e7


def interpolate_heights(point_lat_e7, point_lon_e7, tile_dict, hgt_size):
    """Bilinear interpolation of heights using per-tile lookup.

    The original create_degree() selects a single SRTM tile per grid point
    using floor(lat) and floor(lon), then interpolates within that tile.
    SRTM tiles overlap by one pixel at boundaries and the overlap values
    can disagree, so we must replicate this per-tile selection exactly.

    point_lat_e7, point_lon_e7: shape (n_blocks, 28, 32), int64
    tile_dict: {(lat_int, lon_int): numpy_array or None}
    Returns heights array shape (n_blocks, 28, 32), int16.
    """
    n = hgt_size - 1  # pixels per degree
    shape = point_lat_e7.shape
    heights = np.zeros(shape, dtype=np.int16)

    lat_deg = point_lat_e7.astype(np.float64) * 1e-7
    lon_deg = point_lon_e7.astype(np.float64) * 1e-7

    # Determine which tile each point belongs to
    tile_lat = np.floor(lat_deg).astype(np.int32)
    tile_lon = np.floor(lon_deg).astype(np.int32)

    # Process each unique tile
    unique_tiles = set(zip(tile_lat.ravel(), tile_lon.ravel()))
    for (tlat, tlon) in unique_tiles:
        tile = tile_dict.get((tlat, tlon))
        if tile is None:
            continue  # ocean / missing — heights stay 0

        mask = (tile_lat == tlat) & (tile_lon == tlon)
        if not np.any(mask):
            continue

        # Pixel coordinates within this tile
        cy = (lat_deg[mask] - tlat) * n
        cx = (lon_deg[mask] - tlon) * n

        cy_int = np.floor(cy).astype(np.int32)
        cx_int = np.floor(cx).astype(np.int32)
        cy_frac = cy - cy_int
        cx_frac = cx - cx_int

        cy_int = np.clip(cy_int, 0, n - 1)
        cx_int = np.clip(cx_int, 0, n - 1)

        v00 = tile[cy_int, cx_int].astype(np.float64)
        v10 = tile[cy_int, cx_int + 1].astype(np.float64)
        v01 = tile[cy_int + 1, cx_int].astype(np.float64)
        v11 = tile[cy_int + 1, cx_int + 1].astype(np.float64)

        # Two-step bilinear to match original scalar rounding exactly
        val_x0 = v10 * cx_frac + v00 * (1 - cx_frac)
        val_x1 = v11 * cx_frac + v01 * (1 - cx_frac)
        val = val_x1 * cy_frac + val_x0 * (1 - cy_frac)

        heights[mask] = val.astype(np.int16)

    return heights


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
        # Plus version_minor and padding to fill IO_BLOCK_SIZE
        block = header + h_bytes + trailer
        block += struct.pack('B', 1)  # version_minor at offset 1821
        block += b'\x00' * (IO_BLOCK_SIZE - len(block))

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


def write_dat_gz(outpath, outname, file_buf):
    """Compress and write a DAT file atomically."""
    dat_name = outname[:-3]  # .DAT name for gzip header
    tmp_path = outpath + '.tmp'
    with open(tmp_path, 'wb') as raw_f:
        with gzip.GzipFile(dat_name, 'wb', fileobj=raw_f) as f:
            f.write(file_buf)
    os.rename(tmp_path, outpath)


def process_tile(hgt_file, hgt_map, output_dir, spacing, fmt, tile_idx=None, tile_total=None, overwrite=False):
    """Process a single HGT tile to produce a DAT.gz file."""
    coords = parse_hgt_filename(hgt_file)
    if coords is None:
        print(f"Skipping unrecognised file: {hgt_file}")
        return

    lat_int, lon_int = coords
    outname = dat_filename(lat_int, lon_int)
    outpath = os.path.join(output_dir, outname)
    progress = f"[{tile_idx}/{tile_total}] " if tile_idx is not None else ""

    if os.path.exists(outpath) and not overwrite:
        print(f"{progress}Skipping {outname} (exists)")
        return

    hgt_cache = {}

    # Step 1: Enumerate valid blocks
    valid_blocks, stride = enumerate_valid_blocks(lat_int, lon_int, spacing, fmt)
    if not valid_blocks:
        print(f"{progress}No valid blocks for {os.path.basename(hgt_file)}")
        return

    # Step 2: Load elevation tiles
    tile_dict, hgt_size = load_tile_dict(
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
            chunk_blocks, lat_int, lon_int, spacing, fmt)

        # Step 4: Interpolate heights
        chunk_heights = interpolate_heights(
            point_lat_e7, point_lon_e7, tile_dict, hgt_size)

        all_heights[chunk_start:chunk_end] = chunk_heights

    # Step 5: Pack into DAT file buffer
    file_buf = pack_dat_file(valid_blocks, all_heights, lat_int, lon_int, spacing, fmt)

    # Step 6: Compress and write atomically
    os.makedirs(output_dir, exist_ok=True)
    write_dat_gz(outpath, outname, file_buf)

    print(f"{progress}Generated {outname} ({n_blocks} blocks)")


def process_ocean_tile(args):
    """Generate an all-zero DAT.gz file for an ocean tile."""
    try:
        lat_int, lon_int, output_dir, spacing, fmt, tile_idx, tile_total, overwrite = args
        outname = dat_filename(lat_int, lon_int)
        outpath = os.path.join(output_dir, outname)
        progress = f"[{tile_idx}/{tile_total}] " if tile_idx is not None else ""

        if os.path.exists(outpath) and not overwrite:
            return

        valid_blocks, stride = enumerate_valid_blocks(lat_int, lon_int, spacing, fmt)
        if not valid_blocks:
            return

        n_blocks = len(valid_blocks)
        all_heights = np.zeros((n_blocks, TERRAIN_GRID_BLOCK_SIZE_X,
                                TERRAIN_GRID_BLOCK_SIZE_Y), dtype=np.int16)

        file_buf = pack_dat_file(valid_blocks, all_heights, lat_int, lon_int, spacing, fmt)
        write_dat_gz(outpath, outname, file_buf)

        print(f"{progress}Ocean {outname} ({n_blocks} blocks)")
    except Exception as e:
        print(f"Error generating ocean tile ({lat_int},{lon_int}): {e}")
        import traceback
        traceback.print_exc()


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
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing output files')
    parser.add_argument('--lat-range', type=int, nargs=2, metavar=('MIN', 'MAX'),
                        help='Generate ocean tiles for all longitudes in this latitude range '
                             '(e.g. --lat-range -85 84 for full world coverage)')
    args = parser.parse_args()

    # Scan for HGT files (flat or continent subdirs)
    all_files = glob.glob(os.path.join(args.hgt_dir, '**/*.hgt.zip'), recursive=True)
    hgt_files = []
    for f in all_files:
        coords = parse_hgt_filename(f)
        if coords is not None:
            hgt_files.append((coords, f))
    # Sort by (lat, lon) for predictable processing order
    hgt_files.sort(key=lambda x: x[0])

    if not hgt_files:
        print(f"No .hgt.zip files found in {args.hgt_dir}")
        sys.exit(1)

    total = len(hgt_files)
    print(f"Found {total} HGT files")

    # Build lookup dict: (lat, lon) -> filepath
    hgt_map = {coords: f for coords, f in hgt_files}

    os.makedirs(args.output_dir, exist_ok=True)

    # Process land tiles from HGT data
    work_args = [(f, hgt_map, args.output_dir, args.spacing, "4.1", i + 1, total, args.overwrite)
                 for i, (coords, f) in enumerate(hgt_files)]

    if args.processes <= 1:
        for wa in work_args:
            process_tile_wrapper(wa)
    else:
        with Pool(processes=args.processes) as pool:
            pool.map(process_tile_wrapper, work_args, chunksize=1)

    # Generate ocean tiles for lat/lon combinations not covered by HGT files
    if args.lat_range is not None:
        lat_min, lat_max = args.lat_range
        ocean_work = []
        idx = 0
        for lat in range(lat_min, lat_max + 1):
            for lon in range(-180, 180):
                if (lat, lon) not in hgt_map:
                    idx += 1
                    ocean_work.append((lat, lon, args.output_dir, args.spacing,
                                       "4.1", idx, None, args.overwrite))

        # Fill in total count
        ocean_total = len(ocean_work)
        ocean_work = [(lat, lon, out, sp, fmt, i, ocean_total, ow)
                      for lat, lon, out, sp, fmt, i, _, ow in ocean_work]

        print(f"Generating {ocean_total} ocean tiles...")

        if args.processes <= 1:
            for wa in ocean_work:
                process_ocean_tile(wa)
        else:
            with Pool(processes=args.processes) as pool:
                pool.map(process_ocean_tile, ocean_work, chunksize=4)

    print("Done!")


if __name__ == '__main__':
    main()
