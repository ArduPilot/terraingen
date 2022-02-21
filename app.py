import uuid
import os
import zipfile
import urllib.request
import gzip
from io import BytesIO
import time

from flask import Flask
from flask import render_template
from flask import request

from terrain_gen import add_offset

# Directory of this file
this_path = os.path.dirname(os.path.realpath(__file__))

# Where the user requested tile are stored
output_path = os.path.join(this_path, '..', 'userRequestTerrain')

# Where the data database is
tile_path = os.path.join(this_path, '..', 'data', 'tilesapm41')
url_path = 'https://terrain.ardupilot.org/data/tilesapm41/'

# The output folder for all gzipped terrain requests
app = Flask(__name__, static_url_path='/terrain', static_folder=output_path,)

def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))

def getDatFile(lat, lon):
    '''Get file'''
    if lat < 0:
        NS = 'S'
    else:
        NS = 'N'
    if lon < 0:
        EW = 'W'
    else:
        EW = 'E'
    return "%c%02u%c%03u.DAT.gz" % (NS, min(abs(int(lat)), 99), EW, min(abs(int(lon)), 999))

def compressFiles(fileList, uuidkey, format):
    # create a zip file comprised of dat.gz tiles
    zipthis = os.path.join(output_path, uuidkey + '.zip')

    # create output dirs if needed
    try:
        os.makedirs(output_path)
    except OSError:
        pass
    try:
        os.makedirs(tile_path)
    except OSError:
        pass
        
    try:
        with zipfile.ZipFile(zipthis, 'w') as terrain_zip:
            for fn in fileList:
                if not os.path.exists(fn):
                    #download if required
                    print("Downloading " + os.path.basename(fn))
                    g = urllib.request.urlopen(url_path +
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

    except Exception as ex:
        print("Unexpected error: {0}".format(ex))
        return False

    return True

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['GET', 'POST'])
def generate():
    if request.method == 'POST':
        # parse and sanitise the input
        try:
            # request.form['username']
            lat = float(request.form['lat'])
            lon = float(request.form['long'])
            radius = int(request.form['radius'])
            assert lat < 90
            assert lon < 180
            assert lat > -90
            assert lon > -180
            radius = clamp(radius, 1, 400)
        except:
            print("Bad data")
            return render_template('generate.html', error="Error with input")

        # UUID for this terrain generation
        uuidkey = str(uuid.uuid1())

        # Flag for if user wanted a tile outside +-60deg latitude
        outsideLat = None

        # get a list of files required to cover area
        filelist = []
        done = set()
        for dx in range(-radius, radius):
            for dy in range(-radius, radius):
                (lat2, lon2) = add_offset(lat*1e7, lon*1e7, dx*1000.0, dy*1000.0, format)
                lat_int = int(round(lat2 * 1.0e-7))
                lon_int = int(round(lon2 * 1.0e-7))
                tag = (lat_int, lon_int)
                if tag in done:
                    continue
                done.add(tag)
                # make sure tile is inside the 60deg latitude limit
                if abs(lat_int) <= 60:
                    filelist.append(os.path.join(tile_path, getDatFile(lat_int, lon_int)))
                else:
                    outsideLat = True

        # make sure tile is inside the 60deg latitude limit
        if abs(lat_int) <= 60:
            filelist.append(os.path.join(tile_path, getDatFile(lat_int, lon_int)))
        else:
            outsideLat = True

        # remove duplicates
        filelist = list(dict.fromkeys(filelist))
        print(filelist)

        #compress
        success = compressFiles(filelist, uuidkey, format)

        # as a cleanup, remove any generated terrain older than 24H
        for f in os.listdir(output_path):
            if os.stat(os.path.join(output_path, f)).st_mtime < time.time() - 24 * 60 * 60:
                print("Removing old file: " + str(os.path.join(output_path, f)))
                os.remove(os.path.join(output_path, f))

        if success:
            print("Generated " + "/terrain/" + uuidkey + ".zip")
            return render_template('generate.html', urlkey="/terrain/" + uuidkey + ".zip",
                                   uuidkey=uuidkey, outsideLat=outsideLat)
        else:
            print("Failed " + "/terrain/" + uuidkey + ".zip")
            return render_template('generate.html', error="Cannot generate terrain",
                                   uuidkey=uuidkey)
    else:
        print("Bad get")
        return render_template('generate.html', error="Need to use POST, not GET")

if __name__ == "__main__":
    app.run()
