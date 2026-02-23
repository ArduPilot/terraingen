#!/usr/bin/env python3
"""
Read or set the version_minor field in terrain .DAT and .DAT.gz files.

The version_minor is a uint8 at byte offset 1821 in each 2048-byte grid block.
It is excluded from the CRC calculation for backwards compatibility.

Usage:
    python3 version_minor.py /path/to/tilesdat1          # show version_minor
    python3 version_minor.py /path/to/tilesdat1 --set 1  # set version_minor to 1
    python3 version_minor.py /path/to/tilesdat1 --set 1 --parallel 8
"""

import argparse
import gzip
import os
import struct
import sys
from multiprocessing import Pool

IO_BLOCK_SIZE = 2048
IO_BLOCK_DATA_SIZE = 1821
VERSION_MINOR_OFFSET = 1821

# block header: bitmap(Q=8), lat(i=4), lon(i=4), crc(H=2), version(H=2), spacing(H=2) = 22 bytes
HEADER_FMT = "<QiiHHH"
HEADER_SIZE = struct.calcsize(HEADER_FMT)

TERRAIN_GRID_FORMAT_VERSION = 1


def is_compressed(filepath):
    """Check if a file is gzip compressed."""
    return filepath.endswith('.gz')


def read_data(filepath):
    """Read file contents, decompressing if needed."""
    if is_compressed(filepath):
        with gzip.open(filepath, 'rb') as f:
            return f.read()
    else:
        with open(filepath, 'rb') as f:
            return f.read()


def write_data(filepath, data):
    """Write file contents, compressing if the original was compressed."""
    tmp = filepath + '.tmp'
    if is_compressed(filepath):
        with gzip.open(tmp, 'wb') as f:
            f.write(data)
    else:
        with open(tmp, 'wb') as f:
            f.write(data)
    os.rename(tmp, filepath)


def get_total_blocks(data):
    """Calculate total blocks, handling short last block."""
    size = len(data)
    if (size + IO_BLOCK_SIZE - IO_BLOCK_DATA_SIZE) % IO_BLOCK_SIZE == 0:
        return (size + IO_BLOCK_SIZE - IO_BLOCK_DATA_SIZE) // IO_BLOCK_SIZE
    elif size % IO_BLOCK_SIZE == 0:
        return size // IO_BLOCK_SIZE
    return 0


def is_valid_block(data, block_idx):
    """Check if a block has version==1 (written by terrain generator)."""
    offset = block_idx * IO_BLOCK_SIZE
    if offset + HEADER_SIZE > len(data):
        return False
    bitmap, lat, lon, crc, version, spacing = struct.unpack_from(HEADER_FMT, data, offset)
    return version == TERRAIN_GRID_FORMAT_VERSION


def block_has_trailer(data, block_idx):
    """Check if this block has the trailer (version_minor byte exists)."""
    offset = block_idx * IO_BLOCK_SIZE + VERSION_MINOR_OFFSET
    return offset < len(data)


def read_file(filepath):
    """Read version_minor values from a DAT file. Returns set of values found."""
    data = read_data(filepath)

    total_blocks = get_total_blocks(data)
    if total_blocks == 0:
        return None

    values = set()
    for i in range(total_blocks):
        if is_valid_block(data, i) and block_has_trailer(data, i):
            offset = i * IO_BLOCK_SIZE + VERSION_MINOR_OFFSET
            values.add(struct.unpack_from("B", data, offset)[0])

    return values


def set_file(filepath, value):
    """Set version_minor for all valid blocks in a DAT file."""
    data = bytearray(read_data(filepath))

    total_blocks = get_total_blocks(data)
    if total_blocks == 0:
        return False

    # Pad to full multiple of IO_BLOCK_SIZE if short
    expected_size = total_blocks * IO_BLOCK_SIZE
    if len(data) < expected_size:
        data.extend(b'\x00' * (expected_size - len(data)))

    modified = False
    for i in range(total_blocks):
        if is_valid_block(data, i):
            offset = i * IO_BLOCK_SIZE + VERSION_MINOR_OFFSET
            struct.pack_into("B", data, offset, value)
            modified = True

    if modified:
        # Write to temp file and rename to break hard links
        write_data(filepath, data)

    return modified


def process_read(args):
    """Worker for reading version_minor from a file."""
    filepath, filename = args
    values = read_file(filepath)
    if values is None:
        return f"{filename}: bad file size"
    elif len(values) == 0:
        return f"{filename}: no valid blocks"
    elif len(values) == 1:
        return f"{filename}: version_minor={values.pop()}"
    else:
        return f"{filename}: mixed version_minor={sorted(values)}"


def process_set(args):
    """Worker for setting version_minor in a file."""
    filepath, filename, value = args
    if set_file(filepath, value):
        return f"{filename}: set to {value}"
    else:
        return f"{filename}: no valid blocks"


def main():
    parser = argparse.ArgumentParser(description='Read or set version_minor in terrain DAT files')
    parser.add_argument('directory', help='Directory containing .DAT or .DAT.gz files')
    parser.add_argument('--set', type=int, metavar='N', dest='set_value',
                        help='Set version_minor to N (0-255) for all valid blocks')
    parser.add_argument('--parallel', type=int, metavar='N', default=1,
                        help='Number of parallel workers (default: 1)')
    args = parser.parse_args()

    if args.set_value is not None and not (0 <= args.set_value <= 255):
        print("Error: --set value must be 0-255")
        sys.exit(1)

    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a directory")
        sys.exit(1)

    files = sorted(f for f in os.listdir(args.directory)
                   if f.endswith('.DAT.gz') or f.endswith('.DAT'))
    if not files:
        print(f"No .DAT or .DAT.gz files found in {args.directory}")
        sys.exit(1)

    total = len(files)

    if args.set_value is not None:
        work = [(os.path.join(args.directory, f), f, args.set_value) for f in files]
        if args.parallel > 1:
            with Pool(args.parallel) as pool:
                for i, result in enumerate(pool.imap(process_set, work)):
                    print(f"[{i+1}/{total} {(i+1)*100/total:.1f}%] {result}")
        else:
            for i, w in enumerate(work):
                print(f"[{i+1}/{total} {(i+1)*100/total:.1f}%] {process_set(w)}")
    else:
        work = [(os.path.join(args.directory, f), f) for f in files]
        if args.parallel > 1:
            with Pool(args.parallel) as pool:
                for i, result in enumerate(pool.imap(process_read, work)):
                    print(f"[{i+1}/{total} {(i+1)*100/total:.1f}%] {result}")
        else:
            for i, w in enumerate(work):
                print(f"[{i+1}/{total} {(i+1)*100/total:.1f}%] {process_read(w)}")


if __name__ == '__main__':
    main()
