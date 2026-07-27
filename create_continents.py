#!/usr/bin/env python3
'''
create ardupilot terrain database files as continents

Builds one zip per continent from the current .DAT.gz tiles. Continent
membership comes from the continent subdirectories of an SRTM3 HGT tree
(eg. /mnt/terrain_data/data/SRTM3/Eurasia/*.hgt.zip). Tiles that are in
an existing continent zip but no longer in the HGT tree are kept, and
tiles with no continent (the Misc/ directory) are assigned to the
nearest continent.

Every tile is checked for version_minor >= TERRAIN_VERSION_MINOR_MIN
before it is added; ArduPilot fails the "terrain data expired" pre-arm
check on anything older.
'''
import os
import re
import sys
import gzip
import math
import struct
import zipfile
from argparse import ArgumentParser
from multiprocessing import Pool

IO_BLOCK_SIZE = 2048
VERSION_MINOR_OFFSET = 1821
TERRAIN_GRID_FORMAT_VERSION = 1
# must match TERRAIN_VERSION_MINOR_MIN in ArduPilot's AP_Terrain.h
VERSION_MINOR_MIN = 1

# continent directories that don't name a real continent
NON_CONTINENT = frozenset(['Misc', 'USGS'])

NAME_RE = re.compile(r'^([NS])(\d{2})([EW])(\d{3})')


def tile_latlon(name):
    '''(lat, lon) from a tile name like N47E018'''
    m = NAME_RE.match(name)
    if m is None:
        return None
    lat = int(m.group(2))
    lon = int(m.group(4))
    if m.group(1) == 'S':
        lat = -lat
    if m.group(3) == 'W':
        lon = -lon
    return (lat, lon)


def tile_names(directory):
    '''tile names (no extension) for the .hgt.zip files in a directory'''
    names = set()
    for f in os.listdir(directory):
        if f.endswith('.hgt.zip') and NAME_RE.match(f):
            names.add(f[:-len('.hgt.zip')])
    return names


def hgt_mapping(hgtdir):
    '''map tile name -> continent from an HGT tree with continent subdirs'''
    mapping = {}
    unassigned = set()
    for entry in sorted(os.listdir(hgtdir)):
        subdir = os.path.join(hgtdir, entry)
        if not os.path.isdir(subdir):
            continue
        names = tile_names(subdir)
        if entry in NON_CONTINENT:
            unassigned |= names
        else:
            for n in names:
                mapping[n] = entry
    return mapping, unassigned


def legacy_mapping(contdir):
    '''map tile name -> continent from the existing continent zips'''
    mapping = {}
    if contdir is None or not os.path.isdir(contdir):
        return mapping
    for f in sorted(os.listdir(contdir)):
        if not f.endswith('.zip'):
            continue
        continent = f[:-len('.zip')]
        with zipfile.ZipFile(os.path.join(contdir, f)) as zf:
            for n in zf.namelist():
                n = os.path.basename(n)
                if n.endswith('.DAT'):
                    mapping[n[:-len('.DAT')]] = continent
    return mapping


def placed_tiles(mapping):
    '''sorted (name, continent, lat, lon) for tiles with a known continent'''
    tiles = []
    for name in sorted(mapping):
        pos = tile_latlon(name)
        if pos is not None:
            tiles.append((name, mapping[name], pos[0], pos[1]))
    return tiles


def nearest_continent(name, tiles):
    '''assign a tile to the continent of the closest placed tile'''
    pos = tile_latlon(name)
    if pos is None:
        return None
    best = None
    best_d = None
    for (_, continent, olat, olon) in tiles:
        # longitude wrap: shortest way around the globe
        dlon = abs(pos[1] - olon)
        if dlon > 180:
            dlon = 360 - dlon
        # longitude degrees shrink towards the poles
        dlon *= math.cos(math.radians((pos[0] + olat) * 0.5))
        d = (pos[0] - olat) ** 2 + dlon ** 2
        # tiles are sorted by name, so ties go to the first name
        if best_d is None or d < best_d:
            best_d = d
            best = continent
    return best


def check_version_minor(data):
    '''return the set of version_minor values in a .DAT'''
    values = set()
    for off in range(0, len(data) - VERSION_MINOR_OFFSET, IO_BLOCK_SIZE):
        version = struct.unpack_from('<H', data, off + 18)[0]
        if version == TERRAIN_GRID_FORMAT_VERSION:
            values.add(data[off + VERSION_MINOR_OFFSET])
    return values


def build_continent(job):
    '''build one continent zip, returns (continent, total, done, missing, old)'''
    continent, names, tilesdir, outfolder, dry_run = job

    zipthis = os.path.join(outfolder, continent + '.zip')
    tmp_zipthis = zipthis + '.tmp'

    missing = []
    old = []
    written = 0

    # a dry run reads and checks every tile, it just doesn't write the zip
    terrain_zip = None
    if not dry_run:
        terrain_zip = zipfile.ZipFile(tmp_zipthis, 'w', allowZip64=True)

    try:
        for name in names:
            fn = os.path.join(tilesdir, name + '.DAT.gz')
            if not os.path.exists(fn):
                missing.append(name)
                continue
            with gzip.open(fn, 'rb') as f_in:
                data = f_in.read()
            versions = check_version_minor(data)
            if not versions or min(versions) < VERSION_MINOR_MIN:
                old.append(name)
            if terrain_zip is not None:
                terrain_zip.writestr(name + '.DAT', data,
                                     compress_type=zipfile.ZIP_DEFLATED)
            written += 1
    finally:
        if terrain_zip is not None:
            terrain_zip.close()

    if dry_run:
        return (continent, len(names), written, missing, old)

    if old or missing:
        # publishing a stale tile re-creates the "terrain data expired"
        # pre-arm, and a missing one would silently drop coverage
        os.unlink(tmp_zipthis)
        return (continent, len(names), written, missing, old)

    os.rename(tmp_zipthis, zipthis)
    return (continent, len(names), written, missing, old)


def main():
    parser = ArgumentParser(description='terrain data continent creator')

    parser.add_argument('--tiles-dir', default='/mnt/terrain_data/data/tilesdat3',
                        help='directory of .DAT.gz tiles')
    parser.add_argument('--hgt-dir', default='/mnt/terrain_data/data/SRTM3',
                        help='SRTM3 HGT tree with continent subdirectories')
    parser.add_argument('--out-dir', default='/mnt/terrain_data/data/continentsdat3',
                        help='output directory for the continent zips')
    parser.add_argument('--old-continents', default=None,
                        help='existing continent zip dir, used to keep membership '
                             'of tiles no longer in the HGT tree (defaults to --out-dir)')
    parser.add_argument('--continent', action='append', default=None,
                        help='only build this continent (may be repeated)')
    parser.add_argument('--parallel', type=int, default=1,
                        help='number of continents to build at once')
    parser.add_argument('--dry-run', action='store_true',
                        help='check every tile the build would use, write nothing')

    args = parser.parse_args()

    if args.old_continents is None:
        args.old_continents = args.out_dir

    mapping, unassigned = hgt_mapping(args.hgt_dir)
    print('HGT tree: %u tiles in %u continents, %u unassigned' %
          (len(mapping), len(set(mapping.values())), len(unassigned)))

    # keep tiles that were published before but have since left the HGT tree
    legacy = legacy_mapping(args.old_continents)
    kept = 0
    for name, continent in legacy.items():
        if name not in mapping:
            mapping[name] = continent
            unassigned.discard(name)
            kept += 1
    if legacy:
        print('kept %u tiles only present in the old continent zips' % kept)

    # tiles with no continent go to their nearest neighbour's continent
    base = placed_tiles(mapping)
    for name in sorted(unassigned):
        continent = nearest_continent(name, base)
        if continent is None:
            print('cannot place %s' % name)
            continue
        print('placing %s in %s' % (name, continent))
        mapping[name] = continent

    continents = {}
    for name, continent in mapping.items():
        continents.setdefault(continent, []).append(name)

    if args.continent:
        for c in args.continent:
            if c not in continents:
                print('unknown continent %s, have %s' %
                      (c, ', '.join(sorted(continents))))
                sys.exit(1)
        continents = {c: continents[c] for c in args.continent}

    print('Continents: %s' % ', '.join('%s=%u' % (c, len(continents[c]))
                                       for c in sorted(continents)))
    print('Total tiles: %u' % sum(len(v) for v in continents.values()))

    if not args.dry_run and not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    jobs = [(c, sorted(continents[c]), args.tiles_dir, args.out_dir, args.dry_run)
            for c in sorted(continents, key=lambda c: -len(continents[c]))]

    if args.parallel > 1:
        with Pool(args.parallel) as pool:
            failed = report(pool.imap_unordered(build_continent, jobs), args.dry_run)
    else:
        failed = report((build_continent(j) for j in jobs), args.dry_run)

    sys.exit(1 if failed else 0)


def report(results, dry_run):
    '''print per continent results, return True if any continent failed'''
    failed = False
    for (continent, total, written, missing, old) in results:
        print('%s: %u tiles, %u %s' %
              (continent, total, written, 'checked' if dry_run else 'written'))
        if missing:
            failed = True
            print('  %u tiles missing from tiles dir: %s' %
                  (len(missing), ' '.join(missing[:10])))
        if old:
            failed = True
            print('  %u tiles have version_minor < %u: %s' %
                  (len(old), VERSION_MINOR_MIN, ' '.join(old[:10])))
        if (missing or old) and not dry_run:
            print('  %s.zip NOT written' % continent)
    return failed


if __name__ == '__main__':
    main()
