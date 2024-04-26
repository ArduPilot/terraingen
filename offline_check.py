#!/usr/bin/env python
'''
Check a set of terrain files for corruption
'''
import os
import argparse
import time
import gzip
import shutil
import struct
import fastcrc

crc16 = fastcrc.crc16.xmodem

TERRAIN_GRID_FORMAT_VERSION = 1
IO_BLOCK_SIZE = 2048

# IO block size is 2048
# Actual size is 1821 bytes
# Last 227 bytes is filling
def check_filled(block, lat_int, lon_int, grid_spacing, format):
    '''check a block for validity'''
    if len(block) != IO_BLOCK_SIZE - 227:
        print("Bad size {0} of {1}".format(len(block), IO_BLOCK_SIZE))
        return False
    (bitmap, lat, lon, crc, fileversion, spacing) = struct.unpack("<QiiHHH", block[:22])
    #print(block[:22])
    if(lat == 0 and lon == 0 and crc == 0 and version == 0 and spacing == 0):
        print("Bad block header at block {0}, {1}".format(lat_int, lon_int))
        return False
    if (str(fileversion) != str(TERRAIN_GRID_FORMAT_VERSION)):
        print("Bad version: " + str(fileversion))
        return False
    if abs(lat_int - (lat/1E7)) > 2 or abs(lon_int - (lon/1E7)) > 2:
        print("Bad lat/lon: {0}, {1}".format((lat/1E7), (lon/1E7)))
        return False
    if spacing != grid_spacing:
        print("Bad spacing: %u should be %u" % (spacing, grid_spacing))
        return False
    if bitmap != (1<<56)-1:
        print("Bad bitmap")
        return False

    block = block[:16] + struct.pack("<H", 0) + block[18:]

    crc2 = crc16(block[:1821])
    if crc2 != crc:
        print("Bad CRC")
        return False

    # all is good, return lon/lat of block
    return (lat, lon)

if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description='ArduPilot Terrain DAT file generator')

    # Add the arguments
    # Folder to store processed DAT files
    parser.add_argument('-folder', action="store", dest="folder", default="processedTerrain")
    # File format
    parser.add_argument("--format", type=str, default="4.1", choices=["pre-4.1", "4.1"], help="Ardupilot version")
    # Spacing
    parser.add_argument('--spacing', type=int, default=30)
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    targetFolder = os.path.join(os.getcwd(), args.folder)
    total_errors = 0

    #for each file in folder
    filelist = os.listdir(targetFolder)
    for idx in range(len(filelist)):
        file = filelist[idx]
        print("Checking %s errors=%u %u/%u" % (file, total_errors, idx, len(filelist)))
        if file.endswith("DAT.gz") or file.endswith("DAT"):
            # It's a compressed tile
            # 1. Check it's a valid gzip
            #print(file)
            tile = None
            try:
                lat_int = int(os.path.basename(file)[1:3])
                if os.path.basename(file)[0:1] == "S":
                    lat_int = -lat_int
                lon_int = int(os.path.basename(file)[4:7])
                if os.path.basename(file)[3:4] == "W":
                    lon_int = -lon_int
                if file.endswith("DAT.gz"):
                    with gzip.open(os.path.join(targetFolder, file), 'rb') as f:
                        tile = f.read()
                else:
                    with open(os.path.join(targetFolder, file), 'rb') as f:
                        tile = f.read()
            except Exception as e:
                print("Bad file: " + file)
                print(e)
                total_errors += 1
                continue
            # 2. Is it a valid dat file?
            if (tile):
                
                # 2a. Are the correct number (integer) of blocks present?
                # It will be a multiple of 2048 bytes (block size)
                # There is an missing 227 bytes at the end on the file (2048-1821 = 227), as the 
                # terrain blocks only take up 1821 bytes.
                # APM actually adds the padding on the end, so extra 227 is not needed sometimes
                total_blocks = 0
                if (len(tile)+227) % IO_BLOCK_SIZE == 0:
                    total_blocks = int((len(tile)+227) / IO_BLOCK_SIZE)
                elif len(tile) % IO_BLOCK_SIZE == 0:
                    total_blocks = int(len(tile) / IO_BLOCK_SIZE)
                else:
                    print("Bad file size: {0}. {1} extra bytes at end".format(file, len(tile), len(tile) % IO_BLOCK_SIZE))
                    total_errors += 1

                min_blocks = 450
                max_blocks = 6000
                if abs(lat_int) >= 78:
                    min_blocks = 200
                if args.spacing == 30:
                    min_blocks *= 9
                    max_blocks *= 6000

                if total_blocks > max_blocks or total_blocks < min_blocks:
                    print("Error: %s has %s blocks (range %u to %u)" % (file, total_blocks, min_blocks, max_blocks))
                    total_errors += 1
                # 2b. Does each block have the correct CRC and fields?
                if total_blocks != 0:
                    lat_min = 90 * 1.0e7
                    lat_max = -90 * 1.0e7
                    lon_min = 180 * 1.0e7
                    lon_max = -180 * 1.0e7
                    for blocknum in range(total_blocks):
                        if args.verbose:
                            print("Checking block %u/%u" % (blocknum, total_blocks))
                        block = tile[(blocknum * IO_BLOCK_SIZE):((blocknum + 1)* IO_BLOCK_SIZE)-227]
                        ret = check_filled(block, lat_int, lon_int, args.spacing, args.format)
                        if not ret:
                            print(file)
                            print("Bad data in block {0} of {1}".format(blocknum, total_blocks))
                            total_errors += 1
                            break
                        else:
                            (lat, lon) = ret
                            lat_min = min(lat_min, lat)
                            lat_max = max(lat_max, lat)
                            lon_min = min(lon_min, lon)
                            lon_max = max(lon_max, lon)
                    lat_min *= 1.0e-7
                    lat_max *= 1.0e-7
                    lon_min *= 1.0e-7
                    lon_max *= 1.0e-7
                    lat_range = abs(lat_max-lat_min)
                    lon_range = abs(lon_max-lon_min)
                    allow_lat_range = (0.99, 1.01)
                    allow_lon_range = (1.00, 1.08)
                    if abs(lat_int) > 60:
                        allow_lat_range = (0.99, 1.01)
                        allow_lon_range = (1.00, 1.4)
                    if lat_range < allow_lat_range[0] or lon_range < allow_lon_range[0] or lat_range > allow_lat_range[1] or lon_range > allow_lon_range[1]:
                        print("Bad tile %s has bad range latrange=%.3f lonrange=%.3f" % (file, lat_range, lon_range), allow_lat_range, allow_lon_range)
                        total_errors += 1
                    if args.verbose:
                        print("Tile covers ({0},{1}) to ({2},{3})".format(lat_min, lon_min, lat_max, lon_max))
                        print("Tile size is ({0:.4f}, {1:.4f}) degrees".format(lat_max-lat_min, lon_max-lon_min))
            else:
                print("Bad tile: " + file)
    print("Done!")
