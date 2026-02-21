#!/usr/bin/env python3
"""Create filelist_python for a directory of .hgt.zip files.

The filelist_python is a pickle file used by srtm.py to look up
which HGT files are available and where they are located.
"""

import os
import sys
import re
import pickle
import argparse


def create_filelist(directory):
    """Scan directory for .hgt.zip files and create filelist_python."""
    filename_regex = re.compile(r"([NS])(\d{2})([EW])(\d{3})\.hgt\.zip")
    filelist = {}

    for name in os.listdir(directory):
        match = filename_regex.match(name)
        if match is None:
            continue
        lat = int(match.group(2))
        lon = int(match.group(4))
        if match.group(1) == "S":
            lat = -lat
        if match.group(3) == "W":
            lon = -lon
        filelist[(lat, lon)] = ("/", name)

    outpath = os.path.join(directory, "filelist_python")
    tmppath = outpath + ".tmp"
    with open(tmppath, 'wb') as f:
        pickle.dump(filelist, f)
    os.replace(tmppath, outpath)

    print("Created %s with %u entries" % (outpath, len(filelist)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create filelist_python for an HGT directory")
    parser.add_argument("directory", help="Directory containing .hgt.zip files")
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print("Error: %s is not a directory" % args.directory, file=sys.stderr)
        sys.exit(1)

    create_filelist(args.directory)
