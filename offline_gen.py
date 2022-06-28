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

import srtm
from terrain_gen import create_degree

def worker(downloader, lat, long, targetFolder, startedTiles, totTiles, format, spacing):
    # only create if the output file does not exists
    if os.path.exists(datafile(lat, long, targetFolder) + '.gz'):
        print("Skipping existing compressed tile {0} of {1} ({2:.3f}%)".format(startedTiles+1, totTiles, ((startedTiles)/totTiles)*100))
        return

    if not os.path.exists(datafile(lat, long, targetFolder)):
        if not create_degree(downloader, lat, long, targetFolder, spacing, format):
            #print("Skipping not downloaded tile {0} of {1} ({2:.3f}%)".format(startedTiles+1, totTiles, ((startedTiles)/totTiles)*100))
            time.sleep(0.2)
            return
        print("Created tile {0} of {1}".format(startedTiles, totTiles))
    else:
        print("Skipping existing tile {0} of {1} ({2:.3f}%)".format(startedTiles+1, totTiles, ((startedTiles)/totTiles)*100))

    # and compress
    with open(datafile(lat, long, targetFolder), 'rb') as f_in:
        with gzip.open(datafile(lat, long, targetFolder) + '.gz', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(datafile(lat, long, targetFolder))

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
    global filelistDownloadActive
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
    args = parser.parse_args()

    targetFolder = os.path.join(os.getcwd(), args.folder)
    #create folder if required
    try:
        os.mkdir(targetFolder)
    except FileExistsError:
        pass

    print("Storing in " + targetFolder)

    # store the threads
    processes = []
    
    # Spacing
    if args.database == "SRTM1":
        spacing = 30
    else:
        spacing = 100

    downloader = srtm.SRTMDownloader(debug=False, offline=(1 if args.offline else 0), directory=args.database, cachedir=args.cachedir)
    downloader.loadFileList()
    
    # make tileID's
    tileID = []
    i = 0
    for long in range(-180, 180):
        for lat in range (-(args.latitude+1), args.latitude+1):
            tileID.append([lat, long, i]) 
            i += 1

    # total number of tiles
    totTiles = len(tileID)
    startedTiles = 0

    # Use a pool of workers to process
    with ThreadPool(args.processes-1) as p:
        reslist = [p.apply_async(worker, args=(downloader, td[0], td[1], targetFolder, td[2], len(tileID), args.format, spacing), error_callback=error_handler) for td in tileID]
        for result in reslist:
            result.get()

    print("--------------------------")
    print("All tiles generated!")
    print("--------------------------")

