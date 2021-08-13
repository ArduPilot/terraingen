#!/usr/bin/env python
'''
create ardupilot terrain database files
'''

import math, struct, os
import crc16, time, struct

from MAVProxy.modules.mavproxy_map import srtm

# avoid annoying crc16 DeprecationWarning
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# MAVLink sends 4x4 grids
TERRAIN_GRID_MAVLINK_SIZE = 4

# a 2k grid_block on disk contains 8x7 of the mavlink grids.  Each
# grid block overlaps by one with its neighbour. This ensures that
# the altitude at any point can be calculated from a single grid
# block
TERRAIN_GRID_BLOCK_MUL_X = 7
TERRAIN_GRID_BLOCK_MUL_Y = 8

# this is the spacing between 32x28 grid blocks, in grid_spacing units
TERRAIN_GRID_BLOCK_SPACING_X = ((TERRAIN_GRID_BLOCK_MUL_X-1)*TERRAIN_GRID_MAVLINK_SIZE)
TERRAIN_GRID_BLOCK_SPACING_Y = ((TERRAIN_GRID_BLOCK_MUL_Y-1)*TERRAIN_GRID_MAVLINK_SIZE)

# giving a total grid size of a disk grid_block of 32x28
TERRAIN_GRID_BLOCK_SIZE_X = (TERRAIN_GRID_MAVLINK_SIZE*TERRAIN_GRID_BLOCK_MUL_X)
TERRAIN_GRID_BLOCK_SIZE_Y = (TERRAIN_GRID_MAVLINK_SIZE*TERRAIN_GRID_BLOCK_MUL_Y)

# format of grid on disk
#TERRAIN_GRID_FORMAT_VERSION = 1

IO_BLOCK_SIZE = 2048
IO_BLOCK_DATA_SIZE = 1821
IO_BLOCK_TRAILER_SIZE = IO_BLOCK_SIZE - IO_BLOCK_DATA_SIZE

#GRID_SPACING = 100

def to_float32(f):
    '''emulate single precision float'''
    return struct.unpack('f', struct.pack('f',f))[0]

LOCATION_SCALING_FACTOR = to_float32(0.011131884502145034)
LOCATION_SCALING_FACTOR_INV = to_float32(89.83204953368922)

def longitude_scale(lat):
    '''get longitude scale factor'''
    scale = to_float32(math.cos(to_float32(math.radians(lat))))
    return max(scale, 0.01)

def diff_longitude_E7(lon1, lon2):
    '''get longitude difference, handling wrap'''
    if lon1 * lon2 >= 0:
        # common case of same sign
        return lon1 - lon2
    dlon = lon1 - lon2
    if dlon > 1800000000:
        dlon -= 3600000000
    elif dlon < -1800000000:
        dlon += 3600000000
    return dlon

def get_distance_NE_e7(lat1, lon1, lat2, lon2, version):
    '''get distance tuple between two positions in 1e7 format'''
    if version == 1:
        return ((lat2 - lat1) * LOCATION_SCALING_FACTOR, (lon2 - lon1) * LOCATION_SCALING_FACTOR * longitude_scale(lat1*1.0e-7))
    else:
        dlat = lat2 - lat1
        dlng = diff_longitude_E7(lon2,lon1) * longitude_scale((lat1+lat2)*0.5*1.0e-7)
        return (dlat * LOCATION_SCALING_FACTOR, dlng * LOCATION_SCALING_FACTOR)

def add_offset(lat_e7, lon_e7, ofs_north, ofs_east, version):
    '''add offset in meters to a position'''
    dlat = int(float(ofs_north) * LOCATION_SCALING_FACTOR_INV)
    if version == 1:
        dlng = int((float(ofs_east) * LOCATION_SCALING_FACTOR_INV) / longitude_scale(lat_e7*1.0e-7))
    else:
        dlng = int((float(ofs_east) * LOCATION_SCALING_FACTOR_INV) / longitude_scale((lat_e7+dlat*0.5)*1.0e-7))
    return (int(lat_e7+dlat), int(lon_e7+dlng))

def east_blocks(lat_e7, lon_e7, grid_spacing, version):
    '''work out how many blocks per stride on disk'''
    lat2_e7 = lat_e7
    lon2_e7 = lon_e7 + 10*1000*1000

    # shift another two blocks east to ensure room is available
    lat2_e7, lon2_e7 = add_offset(lat2_e7, lon2_e7, 0, 2*grid_spacing*TERRAIN_GRID_BLOCK_SIZE_Y, version)
    offset = get_distance_NE_e7(lat_e7, lon_e7, lat2_e7, lon2_e7, version)
    return int(offset[1] / (grid_spacing*TERRAIN_GRID_BLOCK_SPACING_Y))

def pos_from_file_offset(lat_degrees, lon_degrees, file_offset, grid_spacing, version):
    '''return a lat/lon in 1e7 format given a file offset'''

    ref_lat = int(lat_degrees*10*1000*1000)
    ref_lon = int(lon_degrees*10*1000*1000)

    stride = east_blocks(ref_lat, ref_lon, grid_spacing, version)
    blocks = file_offset // IO_BLOCK_SIZE
    grid_idx_x = blocks // stride
    grid_idx_y = blocks % stride

    idx_x = grid_idx_x * TERRAIN_GRID_BLOCK_SPACING_X
    idx_y = grid_idx_y * TERRAIN_GRID_BLOCK_SPACING_Y
    offset = (idx_x * grid_spacing, idx_y * grid_spacing)

    (lat_e7, lon_e7) = add_offset(ref_lat, ref_lon, offset[0], offset[1], version)

    offset = get_distance_NE_e7(ref_lat, ref_lon, lat_e7, lon_e7, version)
    grid_idx_x = int(idx_x / TERRAIN_GRID_BLOCK_SPACING_X)
    grid_idx_y = int(idx_y / TERRAIN_GRID_BLOCK_SPACING_Y)

    (lat_e7, lon_e7) = add_offset(ref_lat, ref_lon,
                                  grid_idx_x * TERRAIN_GRID_BLOCK_SPACING_X * float(grid_spacing),
                                  grid_idx_y * TERRAIN_GRID_BLOCK_SPACING_Y * float(grid_spacing),
                                  version)

    return (lat_e7, lon_e7)

class GridBlock(object):
    def __init__(self, lat_int, lon_int, lat, lon, grid_spacing, version):
        '''
        a grid block is a structure in a local file containing height
        information. Each grid block is 2048 bytes in size, to keep file IO to
        block oriented SD cards efficient
        '''

        # crc of whole block, taken with crc=0
        self.crc = 0

        # format version number
        self.version = version

        # grid spacing in meters
        self.spacing = grid_spacing

        # heights in meters over a 32*28 grid
        self.height = []
        for x in range(TERRAIN_GRID_BLOCK_SIZE_X):
            self.height.append([0]*TERRAIN_GRID_BLOCK_SIZE_Y)

        # bitmap of 4x4 grids filled in from GCS (56 bits are used)
        self.bitmap = (1<<56)-1

        lat_e7 = int(lat * 1.0e7)
        lon_e7 = int(lon * 1.0e7)

        # grids start on integer degrees. This makes storing terrain data on
        # the SD card a bit easier. Note that this relies on the python floor
        # behaviour with integer division
        self.lat_degrees = lat_int
        self.lon_degrees = lon_int

        # create reference position for this rounded degree position
        ref_lat = self.lat_degrees*10*1000*1000
        ref_lon = self.lon_degrees*10*1000*1000

        # find offset from reference
        offset = get_distance_NE_e7(ref_lat, ref_lon, lat_e7, lon_e7, self.version)

        offset = (round(offset[0]), round(offset[1]))

        # get indices in terms of grid_spacing elements
        idx_x = int(offset[0] / self.spacing)
        idx_y = int(offset[1] / self.spacing)

        # find indexes into 32*28 grids for this degree reference. Note
        # the use of TERRAIN_GRID_BLOCK_SPACING_{X,Y} which gives a one square
        # overlap between grids
        self.grid_idx_x = idx_x // TERRAIN_GRID_BLOCK_SPACING_X
        self.grid_idx_y = idx_y // TERRAIN_GRID_BLOCK_SPACING_Y

        # calculate lat/lon of SW corner of 32*28 grid_block
        (ref_lat, ref_lon) = add_offset(ref_lat, ref_lon,
                                        self.grid_idx_x * TERRAIN_GRID_BLOCK_SPACING_X * float(self.spacing),
                                        self.grid_idx_y * TERRAIN_GRID_BLOCK_SPACING_Y * float(self.spacing),
                                        self.version)
        self.lat = ref_lat
        self.lon = ref_lon

    def fill(self, gx, gy, altitude):
        '''fill a square'''
        self.height[gx][gy] = int(altitude)

    def blocknum(self):
        '''find IO block number'''
        stride = east_blocks(self.lat_degrees*1e7, self.lon_degrees*1e7, self.spacing, self.version)
        return stride * self.grid_idx_x + self.grid_idx_y

class DataFile(object):
    def __init__(self, lat, lon, folder):
        if lat < 0:
            NS = 'S'
        else:
            NS = 'N'
        if lon < 0:
            EW = 'W'
        else:
            EW = 'E'
        self.name = folder + "/%c%02u%c%03u.DAT" % (NS, min(abs(int(lat)), 99),
                                             EW, min(abs(int(lon)), 999))
        self.tmpname = folder + "/%c%02u%c%03u.DAT.tmp" % (NS, min(abs(int(lat)), 99),
                                             EW, min(abs(int(lon)), 999))
        try:
            os.mkdir(folder)
        except Exception:
            pass
        if not os.path.exists(self.name):
            self.fh = open(self.tmpname, 'w+b')
        else:
            self.fh = open(self.name, 'r+b')

    def finalise(self):
        '''finalise file after writing'''
        self.fh.close()
        #and rename
        os.rename(self.tmpname, self.name)
        
    def remove(self):
        self.fh.close()
        if os.path.exists(self.name):
            os.remove(self.name)
        if os.path.exists(self.tmpname):
            os.remove(self.tmpname)
            
    def seek_offset(self, block):
        '''seek to right offset'''
        # work out how many longitude blocks there are at this latitude
        file_offset = block.blocknum() * IO_BLOCK_SIZE
        self.fh.seek(file_offset)

    def pack(self, block):
        '''pack into a block'''
        buf = bytes()
        buf += struct.pack("<QiiHHH", block.bitmap, block.lat, block.lon, block.crc, block.version, block.spacing)
        for gx in range(TERRAIN_GRID_BLOCK_SIZE_X):
            buf += struct.pack("<%uh" % TERRAIN_GRID_BLOCK_SIZE_Y, *block.height[gx])
        buf += struct.pack("<HHhb", block.grid_idx_x, block.grid_idx_y, block.lon_degrees, block.lat_degrees)
        buf += struct.pack("%uB" % IO_BLOCK_TRAILER_SIZE, *[0]*IO_BLOCK_TRAILER_SIZE)
        return buf

    def write(self, block):
        '''write a grid block'''
        self.seek_offset(block)
        block.crc = 0
        buf = self.pack(block)
        block.crc = crc16.crc16xmodem(buf[:IO_BLOCK_DATA_SIZE])
        buf = self.pack(block)
        self.fh.write(buf)

    def check_filled(self, block, grid_spacing, version):
        '''read a grid block and check if already filled'''
        self.seek_offset(block)
        buf = self.fh.read(IO_BLOCK_SIZE)
        if len(buf) != IO_BLOCK_SIZE:
            return False
        (bitmap, lat, lon, crc, versionfile, spacing) = struct.unpack("<QiiHHH", buf[:22])
        if (versionfile != version or
            abs(lat - block.lat)>2 or
            abs(lon - block.lon)>2 or
            spacing != grid_spacing or
            bitmap != (1<<56)-1):
            return False
        buf = buf[:16] + struct.pack("<H", 0) + buf[18:]
        crc2 = crc16.crc16xmodem(buf[:1821])
        if crc2 != crc:
            return False
        return True

def create_degree(downloader, lat, lon, folder, grid_spacing, version):
    '''create data file for one degree lat/lon'''
    lat_int = int(math.floor(lat))
    lon_int = int(math.floor((lon)))

    tiles = {}

    dfile = DataFile(lat_int, lon_int, folder)

    blocknum = -1

    while True:
        blocknum += 1
        (lat_e7, lon_e7) = pos_from_file_offset(lat_int, lon_int, blocknum * IO_BLOCK_SIZE, grid_spacing, version)
        if lat_e7*1.0e-7 - lat_int >= 1.0:
            break
        lat = lat_e7 * 1.0e-7
        lon = lon_e7 * 1.0e-7
        grid = GridBlock(lat_int, lon_int, lat, lon, grid_spacing, version)
        if grid.blocknum() != blocknum:
            continue
        if dfile.check_filled(grid, grid_spacing, version):
            continue
        for gx in range(TERRAIN_GRID_BLOCK_SIZE_X):
            for gy in range(TERRAIN_GRID_BLOCK_SIZE_Y):
                lat_e7, lon_e7 = add_offset(lat*1.0e7, lon*1.0e7, gx*grid_spacing, gy*grid_spacing, version)
                lat2_int = int(math.floor(lat_e7*1.0e-7))
                lon2_int = int(math.floor(lon_e7*1.0e-7))
                tile_idx = (lat2_int, lon2_int)
                while not tile_idx in tiles:
                    tile = downloader.getTile(lat2_int, lon2_int)
                    waited = False
                    if (tile == 0 and downloader.offline == 1) or (isinstance(tile, srtm.SRTMOceanTile) and downloader.offline == 1):
                        #skip tile
                        dfile.remove()
                        return False
                    elif tile == 0:
                        print("waiting on download of %d,%d" % (lat2_int, lon2_int))
                        time.sleep(0.3)
                        waited = True
                        continue
                    if waited:
                        print("downloaded %d,%d" % (lat2_int, lon2_int))
                    print("Creating for %d %d" % (lat_int, lon_int))
                    tiles[tile_idx] = tile
                if isinstance(tile, srtm.SRTMOceanTile):
                    # if it's a blank ocean tile, there's a quicker way to generate the tile
                    grid.fill(gx, gy, 0)
                else:
                    altitude = tiles[tile_idx].getAltitudeFromLatLon(lat_e7*1.0e-7, lon_e7*1.0e-7)
                    grid.fill(gx, gy, altitude)
        dfile.write(grid)

    dfile.finalise()
    return True

