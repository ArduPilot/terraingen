#!/usr/bin/env python3
'''
Horizontal altitude slice visualiser for comparing DAT and HGT terrain data.
Shows altitude profiles at a given latitude across a terrain tile, useful for
spotting horizontal offsets between ground truth (HGT) and generated (DAT) data.
'''

import argparse
import array
import gzip
import math
import os
import re
import struct
import zipfile

import matplotlib.pyplot as plt

# Constants from terrain_gen.py
TERRAIN_GRID_MAVLINK_SIZE = 4
TERRAIN_GRID_BLOCK_MUL_X = 7
TERRAIN_GRID_BLOCK_MUL_Y = 8
TERRAIN_GRID_BLOCK_SPACING_X = (TERRAIN_GRID_BLOCK_MUL_X - 1) * TERRAIN_GRID_MAVLINK_SIZE  # 24
TERRAIN_GRID_BLOCK_SPACING_Y = (TERRAIN_GRID_BLOCK_MUL_Y - 1) * TERRAIN_GRID_MAVLINK_SIZE  # 28
TERRAIN_GRID_BLOCK_SIZE_X = TERRAIN_GRID_MAVLINK_SIZE * TERRAIN_GRID_BLOCK_MUL_X   # 28
TERRAIN_GRID_BLOCK_SIZE_Y = TERRAIN_GRID_MAVLINK_SIZE * TERRAIN_GRID_BLOCK_MUL_Y   # 32
IO_BLOCK_SIZE = 2048


def to_float32(f):
    '''Emulate single precision float'''
    return struct.unpack('f', struct.pack('f', f))[0]


LOCATION_SCALING_FACTOR = to_float32(0.011131884502145034)
LOCATION_SCALING_FACTOR_INV = to_float32(89.83204953368922)


def longitude_scale(lat):
    '''Get longitude scale factor'''
    scale = to_float32(math.cos(to_float32(math.radians(lat))))
    return max(scale, 0.01)


def diff_longitude_E7(lon1, lon2):
    '''Get longitude difference, handling wrap'''
    if lon1 * lon2 >= 0:
        return lon1 - lon2
    dlon = lon1 - lon2
    if dlon > 1800000000:
        dlon -= 3600000000
    elif dlon < -1800000000:
        dlon += 3600000000
    return dlon


def get_distance_NE_e7(lat1, lon1, lat2, lon2, fmt="4.1"):
    '''Get distance tuple between two positions in 1e7 format'''
    if fmt == "pre-4.1":
        return ((lat2 - lat1) * LOCATION_SCALING_FACTOR,
                (lon2 - lon1) * LOCATION_SCALING_FACTOR * longitude_scale(lat1 * 1.0e-7))
    dlat = lat2 - lat1
    dlng = diff_longitude_E7(lon2, lon1) * longitude_scale((lat1 + lat2) * 0.5 * 1.0e-7)
    return (dlat * LOCATION_SCALING_FACTOR, dlng * LOCATION_SCALING_FACTOR)


def parse_tile_name(filename):
    '''Extract lat/lon from filename like N83W033.DAT.gz or N83W033.hgt.zip'''
    basename = os.path.basename(filename)
    match = re.match(r'([NS])(\d{2})([EW])(\d{3})', basename)
    if not match:
        raise ValueError("Cannot parse tile name from %s" % filename)
    lat = int(match.group(2))
    lon = int(match.group(4))
    if match.group(1) == 'S':
        lat = -lat
    if match.group(3) == 'W':
        lon = -lon
    return lat, lon


def read_dat_blocks(filepath):
    '''Read all blocks from a DAT or DAT.gz file.
    Returns (blocks_dict, tile_lat, tile_lon, spacing) where
    blocks_dict maps (grid_idx_x, grid_idx_y) to height[BLOCK_SIZE_X][BLOCK_SIZE_Y]
    '''
    if filepath.endswith('.gz'):
        with gzip.open(filepath, 'rb') as f:
            data = f.read()
    else:
        with open(filepath, 'rb') as f:
            data = f.read()

    blocks = {}
    tile_lat = None
    tile_lon = None
    spacing = None

    num_blocks = len(data) // IO_BLOCK_SIZE
    for i in range(num_blocks):
        block_data = data[i * IO_BLOCK_SIZE:(i + 1) * IO_BLOCK_SIZE]
        if len(block_data) < IO_BLOCK_SIZE:
            break

        # Parse header: bitmap(Q) lat(i) lon(i) crc(H) version(H) spacing(H)
        bitmap, lat, lon, crc, version, sp = struct.unpack("<QiiHHH", block_data[:22])
        if version == 0 and sp == 0:
            continue  # empty block

        # Parse heights: BLOCK_SIZE_X rows of BLOCK_SIZE_Y int16
        heights = []
        offset = 22
        for gx in range(TERRAIN_GRID_BLOCK_SIZE_X):
            row = struct.unpack("<%dh" % TERRAIN_GRID_BLOCK_SIZE_Y,
                                block_data[offset:offset + TERRAIN_GRID_BLOCK_SIZE_Y * 2])
            heights.append(list(row))
            offset += TERRAIN_GRID_BLOCK_SIZE_Y * 2

        # Parse trailer: grid_idx_x(H) grid_idx_y(H) lon_degrees(h) lat_degrees(b)
        grid_idx_x, grid_idx_y, lon_deg, lat_deg = struct.unpack("<HHhb", block_data[offset:offset + 7])

        if tile_lat is None:
            tile_lat = lat_deg
            tile_lon = lon_deg
            spacing = sp

        blocks[(grid_idx_x, grid_idx_y)] = heights

    return blocks, tile_lat, tile_lon, spacing


def read_hgt_data(filepath):
    '''Read HGT data from .hgt.zip file.
    Returns (data_array, size) where size is 1201 or 3601.
    '''
    with zipfile.ZipFile(filepath, 'r') as zf:
        names = zf.namelist()
        raw = zf.read(names[0])

    size = int(math.sqrt(len(raw) / 2))
    data = array.array('h', raw)
    data.byteswap()
    return data, size


def dat_altitude(lat, lon, blocks, tile_lat, tile_lon, spacing, fmt="4.1"):
    '''Compute altitude from DAT data using AP_Terrain interpolation.
    Single-step get_distance_NE from degree corner, matching vehicle code.
    '''
    ref_lat = int(tile_lat * 1e7)
    ref_lon = int(tile_lon * 1e7)
    point_lat = int(lat * 1e7)
    point_lon = int(lon * 1e7)

    offset = get_distance_NE_e7(ref_lat, ref_lon, point_lat, point_lon, fmt)
    offset_north = offset[0]
    offset_east = offset[1]

    idx_x = int(offset_north / spacing)
    idx_y = int(offset_east / spacing)

    grid_idx_x = idx_x // TERRAIN_GRID_BLOCK_SPACING_X
    grid_idx_y = idx_y // TERRAIN_GRID_BLOCK_SPACING_Y

    local_x = idx_x % TERRAIN_GRID_BLOCK_SPACING_X
    local_y = idx_y % TERRAIN_GRID_BLOCK_SPACING_Y

    frac_x = (offset_north - idx_x * spacing) / spacing
    frac_y = (offset_east - idx_y * spacing) / spacing

    key = (grid_idx_x, grid_idx_y)
    if key not in blocks:
        return None

    heights = blocks[key]

    if local_x < 0 or local_x + 1 >= TERRAIN_GRID_BLOCK_SIZE_X:
        return None
    if local_y < 0 or local_y + 1 >= TERRAIN_GRID_BLOCK_SIZE_Y:
        return None

    h00 = heights[local_x][local_y]
    h01 = heights[local_x][local_y + 1]
    h10 = heights[local_x + 1][local_y]
    h11 = heights[local_x + 1][local_y + 1]

    avg1 = (1 - frac_x) * h00 + frac_x * h10
    avg2 = (1 - frac_x) * h01 + frac_x * h11
    result = (1 - frac_y) * avg1 + frac_y * avg2

    return result


def hgt_altitude(lat, lon, hgt_data, tile_lat, tile_lon, hgt_size):
    '''Compute altitude from HGT data using srtm.py interpolation.'''
    lat_norm = lat - tile_lat
    lon_norm = lon - tile_lon

    if lat_norm < 0 or lat_norm >= 1 or lon_norm < 0 or lon_norm >= 1:
        return None

    x = lon_norm * (hgt_size - 1)
    y = lat_norm * (hgt_size - 1)

    x_int = int(x)
    x_frac = x - x_int
    y_int = int(y)
    y_frac = y - y_int

    def get_pixel(px, py):
        if px >= hgt_size or py >= hgt_size:
            return None
        offset = px + hgt_size * (hgt_size - py - 1)
        value = hgt_data[offset]
        if value == -32768:
            return -1
        return value

    def avg(v1, v2, w):
        if v1 is None:
            return v2
        if v2 is None:
            return v1
        return v2 * w + v1 * (1 - w)

    v00 = get_pixel(x_int, y_int)
    v10 = get_pixel(x_int + 1, y_int)
    v01 = get_pixel(x_int, y_int + 1)
    v11 = get_pixel(x_int + 1, y_int + 1)

    v1 = avg(v00, v10, x_frac)
    v2 = avg(v01, v11, x_frac)
    result = avg(v1, v2, y_frac)

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Horizontal altitude slice visualiser for DAT and HGT terrain data')
    parser.add_argument('files', nargs='+', help='.DAT.gz and/or .hgt.zip files for the same tile')
    parser.add_argument('--percent', type=float, default=50,
                        help='Latitude as percentage through tile (0=south, 100=north)')
    parser.add_argument('--step', type=float, default=10,
                        help='Horizontal sample step in metres')
    parser.add_argument('--format', type=str, default='4.1', choices=['4.1', 'pre-4.1'],
                        help='ArduPilot coordinate format version')
    parser.add_argument('-o', type=str, default=None,
                        help='Save plot to file instead of showing interactively')
    args = parser.parse_args()

    # Determine tile lat/lon from first file
    tile_lat, tile_lon = parse_tile_name(args.files[0])

    # Compute target latitude from percentage
    target_lat = tile_lat + args.percent / 100.0

    # Generate longitude sample points
    lon_step_deg = args.step / (111320.0 * math.cos(math.radians(target_lat)))
    lons = []
    lon = float(tile_lon)
    while lon < tile_lon + 1.0:
        lons.append(lon)
        lon += lon_step_deg

    print("Tile: %d,%d  Target lat: %.6f  Samples: %d" % (tile_lat, tile_lon, target_lat, len(lons)))

    fig, ax = plt.subplots(figsize=(14, 6))

    for filepath in args.files:
        label = os.path.basename(filepath)
        alts = []

        if '.DAT' in filepath:
            blocks, t_lat, t_lon, spacing = read_dat_blocks(filepath)
            print("  %s: %d blocks, spacing=%dm" % (label, len(blocks), spacing))
            for lon_val in lons:
                alt = dat_altitude(target_lat, lon_val, blocks, t_lat, t_lon, spacing, args.format)
                alts.append(alt)
        elif '.hgt' in filepath.lower():
            file_lat, file_lon = parse_tile_name(filepath)
            hgt_data, hgt_size = read_hgt_data(filepath)
            print("  %s: %dx%d pixels" % (label, hgt_size, hgt_size))
            for lon_val in lons:
                alt = hgt_altitude(target_lat, lon_val, hgt_data, file_lat, file_lon, hgt_size)
                alts.append(alt)
        else:
            print("  Skipping unrecognised file: %s" % filepath)
            continue

        # Filter out None values for plotting
        plot_lons = [lons[i] for i in range(len(alts)) if alts[i] is not None]
        plot_alts = [a for a in alts if a is not None]
        ax.plot(plot_lons, plot_alts, label=label, linewidth=0.8)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Altitude (m)')
    ax.set_title('Altitude slice at lat=%.4f (%g%% through tile)' % (target_lat, args.percent))
    ax.legend()
    ax.grid(True, alpha=0.3)

    if args.o:
        fig.savefig(args.o, dpi=150, bbox_inches='tight')
        print("Saved to %s" % args.o)
    else:
        plt.show()


if __name__ == '__main__':
    main()
