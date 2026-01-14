#!/usr/bin/env python3
'''
Generation of all dat files at 100m spacing.
Preprocessing so the website doesn't have to

This will take a long time to process!
'''
import os
#from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
import argparse
import time
import gzip
import shutil
import struct
import threading
import math
from datetime import datetime

import srtm
from terrain_gen import create_degree, add_offset

# Thread-safe logging for --regen-ocean
regen_log_lock = threading.Lock()
regen_log_file = None

# Date when the ocean tile bug was fixed (used for --regen-ocean)
# Tiles generated before this date may be affected
OCEAN_BUG_FIX_DATE = time.mktime(time.strptime("2026-01-12", "%Y-%m-%d"))

def is_tile_affected(filepath):
    """Check if a DAT.gz file could be affected by the ocean tile bug.

    The bug caused land cells to get zero altitude when processed after ocean tiles.
    Any tile with both zeros and non-zeros could potentially be affected.
    Pure ocean (all zeros) and pure land (no zeros) tiles are not affected.

    Returns True if the tile should be regenerated, False to skip.
    """
    # Skip files generated after the fix
    try:
        if os.path.getmtime(filepath) > OCEAN_BUG_FIX_DATE:
            return False
    except OSError:
        return False

    try:
        # Read and decompress
        with gzip.open(filepath, 'rb') as f:
            data = f.read()

        # Count non-zero heights across all blocks
        IO_BLOCK_SIZE = 2048
        total_heights = 0
        nonzero_count = 0
        num_blocks = len(data) // IO_BLOCK_SIZE

        for block_idx in range(num_blocks):
            block_start = block_idx * IO_BLOCK_SIZE
            # Heights start at offset 22, 896 heights * 2 bytes each
            height_data = data[block_start + 22:block_start + 22 + 1792]
            if len(height_data) < 1792:
                continue
            heights = struct.unpack('<896h', height_data)
            total_heights += 896
            nonzero_count += sum(1 for h in heights if h != 0)

        if total_heights == 0:
            return False

        # Any tile with both zeros and non-zeros could be affected
        # Pure ocean (all zeros): not affected - no land to corrupt
        # Pure land (no zeros): not affected - no ocean tile lookups
        # Mixed (some zeros, some non-zeros): could be affected
        has_zeros = nonzero_count < total_heights
        has_nonzeros = nonzero_count > 0
        return has_zeros and has_nonzeros

    except Exception as e:
        print(f"Error checking {filepath}: {e}")
        return False

def read_tile_heights(filepath):
    """Read all heights from a DAT or DAT.gz file.

    Returns a list of all height values, or None on error.
    """
    try:
        if filepath.endswith('.gz'):
            with gzip.open(filepath, 'rb') as f:
                data = f.read()
        else:
            with open(filepath, 'rb') as f:
                data = f.read()

        IO_BLOCK_SIZE = 2048
        all_heights = []
        num_blocks = len(data) // IO_BLOCK_SIZE

        for block_idx in range(num_blocks):
            block_start = block_idx * IO_BLOCK_SIZE
            height_data = data[block_start + 22:block_start + 22 + 1792]
            if len(height_data) < 1792:
                continue
            heights = struct.unpack('<896h', height_data)
            all_heights.extend(heights)

        return all_heights
    except Exception as e:
        print(f"Error reading heights from {filepath}: {e}")
        return None

def compare_tile_heights(old_heights, new_heights):
    """Compare old and new height data.

    Returns a dict with statistics about the changes.
    """
    if old_heights is None or new_heights is None:
        return None
    if len(old_heights) != len(new_heights):
        return None

    GRID_SIZE_X = 32
    GRID_SIZE_Y = 28
    HEIGHTS_PER_BLOCK = GRID_SIZE_X * GRID_SIZE_Y

    total_points = len(old_heights)
    changed_count = 0
    zeros_to_nonzero = 0  # Bug-affected points (was 0, now has real altitude)
    max_bug_altitude = 0  # Max altitude among bug-affected points
    max_bug_altitude_idx = 0
    # Also track best "isolated" point (surrounded by zeros) for clearer demonstration
    best_isolated_altitude = 0
    best_isolated_idx = 0

    for idx, (old_h, new_h) in enumerate(zip(old_heights, new_heights)):
        if old_h != new_h:
            changed_count += 1
            if old_h == 0 and new_h > 0:
                zeros_to_nonzero += 1
                # Track the highest altitude that was incorrectly zero
                if new_h > max_bug_altitude:
                    max_bug_altitude = new_h
                    max_bug_altitude_idx = idx

                # Check if this point is "isolated" (neighbors also zero in old data)
                # This makes for a clearer demonstration since terrain queries interpolate
                block_num = idx // HEIGHTS_PER_BLOCK
                pos_in_block = idx % HEIGHTS_PER_BLOCK
                gx = pos_in_block // GRID_SIZE_Y
                gy = pos_in_block % GRID_SIZE_Y

                if 1 <= gx < 31 and 1 <= gy < 27:  # Not on edge
                    block_base = block_num * HEIGHTS_PER_BLOCK
                    neighbors_zero = True
                    for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                        nidx = block_base + (gx+dx) * GRID_SIZE_Y + (gy+dy)
                        if nidx < len(old_heights) and old_heights[nidx] != 0:
                            neighbors_zero = False
                            break
                    if neighbors_zero and new_h > best_isolated_altitude:
                        best_isolated_altitude = new_h
                        best_isolated_idx = idx

    # Prefer isolated point for location if available, otherwise use max
    if best_isolated_altitude > 0:
        report_altitude = best_isolated_altitude
        report_idx = best_isolated_idx
    else:
        report_altitude = max_bug_altitude
        report_idx = max_bug_altitude_idx

    return {
        'total_points': total_points,
        'changed_count': changed_count,
        'changed_pct': (changed_count / total_points * 100) if total_points > 0 else 0,
        'zeros_to_nonzero': zeros_to_nonzero,
        'zeros_to_nonzero_pct': (zeros_to_nonzero / total_points * 100) if total_points > 0 else 0,
        'max_bug_altitude': max_bug_altitude,
        'report_altitude': report_altitude,
        'report_idx': report_idx
    }

def index_to_latlon(filepath, idx, spacing):
    """Convert a height index to lat/lon coordinates.

    Each block has 896 heights (32x28 grid). The block header contains
    the lat/lon of the block's SW corner in 1e7 format.
    """
    try:
        if filepath.endswith('.gz'):
            with gzip.open(filepath, 'rb') as f:
                data = f.read()
        else:
            with open(filepath, 'rb') as f:
                data = f.read()

        IO_BLOCK_SIZE = 2048
        GRID_SIZE_X = 32
        GRID_SIZE_Y = 28
        HEIGHTS_PER_BLOCK = GRID_SIZE_X * GRID_SIZE_Y  # 896

        block_num = idx // HEIGHTS_PER_BLOCK
        pos_in_block = idx % HEIGHTS_PER_BLOCK

        # Heights are stored column-major: gx varies slower
        gx = pos_in_block // GRID_SIZE_Y
        gy = pos_in_block % GRID_SIZE_Y

        # Read block header to get base lat/lon
        block_start = block_num * IO_BLOCK_SIZE
        if block_start + 22 > len(data):
            return None, None

        # Header: bitmap(8) + lat(4) + lon(4) + ...
        header = data[block_start:block_start + 16]
        (bitmap, lat_e7, lon_e7) = struct.unpack("<Qii", header)

        # Use the same offset calculation as terrain_gen.py
        lat_e7_out, lon_e7_out = add_offset(lat_e7, lon_e7, gx * spacing, gy * spacing, "4.1")

        return lat_e7_out * 1e-7, lon_e7_out * 1e-7
    except Exception as e:
        print(f"Error converting index to lat/lon: {e}")
        return None, None

def log_regen(message):
    """Thread-safe logging to regen-ocean.log"""
    global regen_log_file
    if regen_log_file is None:
        return
    with regen_log_lock:
        regen_log_file.write(message + "\n")
        regen_log_file.flush()

def worker(downloader, lat, long, targetFolder, startedTiles, totTiles, format, spacing, regen_ocean=False, regen_list=False):
    gz_file = datafile(lat, long, targetFolder) + '.gz'
    dat_file = datafile(lat, long, targetFolder)
    old_heights = None
    backup_file = None

    # Check if we need to regenerate due to ocean tile bug
    if regen_ocean:
        if os.path.exists(gz_file):
            if is_tile_affected(gz_file):
                print("Regenerating ocean-bug-affected tile {0} of {1}: {2}".format(
                    startedTiles+1, totTiles, os.path.basename(gz_file)))
                # Read old heights for comparison (don't delete yet!)
                old_heights = read_tile_heights(gz_file)
                # Rename to backup - only delete after successful regeneration
                backup_file = gz_file + '.bak'
                os.rename(gz_file, backup_file)
            else:
                # Tile is not affected, skip it
                return
        elif not os.path.exists(dat_file):
            # No file exists, nothing to regenerate
            return
    elif regen_list:
        # Regenerate from list: don't check if affected, just regenerate
        if os.path.exists(gz_file):
            print("Regenerating from list tile {0} of {1}: {2}".format(
                startedTiles+1, totTiles, os.path.basename(gz_file)))
            # Rename to backup - only delete after successful regeneration
            backup_file = gz_file + '.bak'
            os.rename(gz_file, backup_file)
        elif os.path.exists(dat_file):
            print("Regenerating from list tile {0} of {1}: {2}".format(
                startedTiles+1, totTiles, os.path.basename(dat_file)))
            # Remove uncompressed file to force regeneration
            os.remove(dat_file)
        else:
            # No existing file, just generate fresh
            print("Generating missing tile {0} of {1}: {2}".format(
                startedTiles+1, totTiles, os.path.basename(gz_file)))
    else:
        # Normal mode: skip existing compressed tiles
        if os.path.exists(gz_file):
            print("Skipping existing compressed tile {0} of {1} ({2:.3f}%)".format(startedTiles+1, totTiles, ((startedTiles)/totTiles)*100))
            return

    if not os.path.exists(dat_file):
        if not create_degree(downloader, lat, long, targetFolder, spacing, format):
            #print("Skipping not downloaded tile {0} of {1} ({2:.3f}%)".format(startedTiles+1, totTiles, ((startedTiles)/totTiles)*100))
            # Restore backup if regeneration failed
            if backup_file and os.path.exists(backup_file):
                os.rename(backup_file, gz_file)
                print("Regeneration failed, restored backup: {0}".format(os.path.basename(gz_file)))
            time.sleep(0.2)
            return
        print("Created tile {0} of {1}".format(startedTiles, totTiles))
    else:
        print("Skipping existing tile {0} of {1} ({2:.3f}%)".format(startedTiles+1, totTiles, ((startedTiles)/totTiles)*100))

    # and compress
    with open(dat_file, 'rb') as f_in:
        with gzip.open(gz_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(dat_file)

    # Remove backup after successful regeneration
    if backup_file and os.path.exists(backup_file):
        os.remove(backup_file)

    # Log comparison for --regen-ocean mode
    if regen_ocean and old_heights is not None:
        new_heights = read_tile_heights(gz_file)
        stats = compare_tile_heights(old_heights, new_heights)
        if stats and stats['zeros_to_nonzero'] > 0:
            tile_name = os.path.basename(gz_file)
            rep_lat, rep_lon = index_to_latlon(gz_file, stats['report_idx'], spacing)
            if rep_lat is not None:
                loc_str = f", example: {stats['report_altitude']}m at {rep_lat:.6f},{rep_lon:.6f}"
            else:
                loc_str = ""
            log_regen(f"{tile_name}: {stats['zeros_to_nonzero_pct']:.2f}% bug-affected ({stats['zeros_to_nonzero']}/{stats['total_points']} points), "
                      f"max corrected altitude {stats['max_bug_altitude']}m{loc_str}")
        elif stats and stats['changed_count'] > 0:
            tile_name = os.path.basename(gz_file)
            log_regen(f"{tile_name}: {stats['changed_count']} points changed but none were zero->nonzero")
        elif stats:
            tile_name = os.path.basename(gz_file)
            log_regen(f"{tile_name}: no changes (tile was not actually affected)")

    print("Done tile {0} of {1} ({2:.3f}%)".format(startedTiles+1, totTiles, ((startedTiles+1)/totTiles)*100))
    print("Folder is {0:.0f}Mb in size".format(get_size(targetFolder)/(1024*1024)))

def datafile(lat, lon, folder):
    if lat < 0:
        NS = 'S'
    else:
        NS = 'N'
    if lon < 0:
        EW = 'W'
    else:
        EW = 'E'
    name = folder + "/%c%02u%c%03u.DAT" % (NS, min(abs(int(lat)), 99),
                                    EW, min(abs(int(lon)), 999))

    return name

def parse_tile_name(name):
    """Parse a tile name like 'N00E006' or 'S45W067' to (lat, lon) tuple.

    Returns (lat, lon) or None if invalid format.
    """
    name = name.strip()
    if len(name) < 7:
        return None

    try:
        ns = name[0].upper()
        lat = int(name[1:3])
        ew = name[3].upper()
        lon = int(name[4:7])

        if ns == 'S':
            lat = -lat
        elif ns != 'N':
            return None

        if ew == 'W':
            lon = -lon
        elif ew != 'E':
            return None

        return (lat, lon)
    except (ValueError, IndexError):
        return None

def load_regen_list(filepath):
    """Load list of tile names from file and return as set of (lat, lon) tuples."""
    tiles = set()
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            result = parse_tile_name(line)
            if result:
                tiles.add(result)
            else:
                print(f"Warning: Could not parse tile name: {line}")
    return tiles

def get_size(start_path = '.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                # other threads may be moving files whilst we're getting folder
                # size here
                try:
                    total_size += os.path.getsize(fp)
                except FileNotFoundError:
                    pass

    return total_size

def error_handler(e):
    print('Worker exception: {0}'.format(e))
    
if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description='ArduPilot Terrain DAT file generator')

    # Add the arguments
    # Folder to store processed DAT files
    parser.add_argument('-folder', action="store", dest="folder", default="processedTerrain")
    # Folder to store processed DAT files
    parser.add_argument('--offline', action="store_true", dest="offline")
    # Number of threads to use
    parser.add_argument('-processes', action="store", dest="processes", type=int, default=4)
    # Latitude range
    parser.add_argument('-latitude', action="store", dest="latitude", type=int, default=60)
    # File format
    parser.add_argument("--format", type=str, default="4.1", choices=["pre-4.1", "4.1"], help="Ardupilot version")
    # Source database
    parser.add_argument("--database", type=str, default="SRTM1", choices=["SRTM1", "SRTM3"])
    # Cache dir
    parser.add_argument('--cachedir', action="store", default=None)
    # Regenerate tiles affected by ocean tile bug
    parser.add_argument('--regen-ocean', action="store_true", dest="regen_ocean",
                        help="Only regenerate tiles affected by the ocean tile bug (coastal tiles with incorrect zero heights)")
    # Regenerate specific tiles from a list file
    parser.add_argument('--regen-list', action="store", dest="regen_list",
                        help="File containing list of tile names to regenerate (e.g., N00E006, one per line)")
    args = parser.parse_args()

    targetFolder = os.path.join(os.getcwd(), args.folder)
    #create folder if required
    try:
        os.mkdir(targetFolder)
    except FileExistsError:
        pass

    print("Storing in " + targetFolder)

    # Open log file for --regen-ocean mode
    if args.regen_ocean:
        regen_log_file = open("regen-ocean.log", "a")
        regen_log_file.write(f"\n{'='*60}\n")
        regen_log_file.write(f"Regeneration started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        regen_log_file.write(f"Target folder: {targetFolder}\n")
        regen_log_file.write(f"Database: {args.database}\n")
        regen_log_file.write(f"{'='*60}\n")
        regen_log_file.flush()

    # store the threads
    processes = []
    
    # Spacing
    if args.database == "SRTM1":
        spacing = 30
    else:
        spacing = 100

    downloader = srtm.SRTMDownloader(debug=False, offline=(1 if args.offline else 0), directory=args.database, cachedir=args.cachedir)
    downloader.loadFileList()

    # Load regen list if provided
    regen_list_tiles = None
    if args.regen_list:
        regen_list_tiles = load_regen_list(args.regen_list)
        print(f"Loaded {len(regen_list_tiles)} tiles from {args.regen_list}")

    # make tileID's
    tileID = []
    i = 0
    for long in range(-180, 180):
        for lat in range (-(args.latitude+1), args.latitude+1):
            # If using --regen-list, only include tiles in the list
            if regen_list_tiles is not None:
                if (lat, long) not in regen_list_tiles:
                    continue
            tileID.append([lat, long, i])
            i += 1

    # total number of tiles
    totTiles = len(tileID)
    startedTiles = 0

    # Use a pool of workers to process
    with ThreadPool(args.processes-1) as p:
        reslist = [p.apply_async(worker, args=(downloader, td[0], td[1], targetFolder, td[2], len(tileID), args.format, spacing, args.regen_ocean, args.regen_list is not None), error_callback=error_handler) for td in tileID]
        for result in reslist:
            result.get()

    # Close log file for --regen-ocean mode
    if args.regen_ocean and regen_log_file:
        regen_log_file.write(f"Regeneration completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        regen_log_file.close()
        print("Regeneration log written to regen-ocean.log")

    print("--------------------------")
    print("All tiles generated!")
    print("--------------------------")

