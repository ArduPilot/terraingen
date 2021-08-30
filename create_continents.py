#!/usr/bin/env python3
'''
create ardupilot terrain database files as continents
'''
import os
import zipfile
import gzip
from io import BytesIO

from MAVProxy.modules.mavproxy_map import srtm
from argparse import ArgumentParser

def compressFiles(fileList, continent, outfolder):
    # create a zip file comprised of dat.gz tiles
    zipthis = os.path.join(outfolder, continent + ".zip")
        
    with zipfile.ZipFile(zipthis, 'w') as terrain_zip:
        for fn in fileList:
            if not os.path.exists(fn):
                #download if required
                print("Downloading " + os.path.basename(fn))
                g = urllib.request.urlopen('https://terrain.ardupilot.org/data/tilesapm41/' +
                                           os.path.basename(fn))
                print("Downloaded " + os.path.basename(fn))
                with open(fn, 'b+w') as f:
                    f.write(g.read())

            # need to decompress file and pass to zip
            with gzip.open(fn, 'r') as f_in:
                myio = BytesIO(f_in.read())
                print("Decomp " + os.path.basename(fn))

                # and add file to zip
                terrain_zip.writestr(os.path.basename(fn)[:-3], myio.read(),
                                     compress_type=zipfile.ZIP_DEFLATED)

    return True

if __name__ == '__main__':
    this_path = os.path.dirname(os.path.realpath(__file__))

    parser = ArgumentParser(description='terrain data continent creator')

    parser.add_argument("infolder", type=str, default="./files")
    parser.add_argument("outfolder", type=str, default="./continents")

    args = parser.parse_args()

    downloader = srtm.SRTMDownloader(debug=False)
    downloader.loadFileList()

    continents = {}

    # Create mappings of long/lat to continent
    for (lonlat, contfile) in downloader.filelist.items():
        continent = str(contfile[0][:-1])
        filename = os.path.join(this_path, args.infolder, str(contfile[1]).split(".")[0] + ".DAT.gz")
        print(continent + " in " + filename)

        # add to database
        if continent != '' and continent != 'USGS':
            if continent in continents:
                continents[continent].append(filename)
            else:
                continents[continent] = [filename]

    print("Continents are: " + str(continents.keys()))

    # Add the files
    for continent in continents:
        if os.path.exists(os.path.join(args.outfolder, continent + ".zip")):
            os.remove(os.path.join(args.outfolder, continent + ".zip"))
        print("Processing: " + continent)
        files = continents[continent]
        compressFiles(files, continent, args.outfolder)
