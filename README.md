# ArduPilot terrain generator

## Summary

This is a website that pre-generates terrain files for Ardupilot. The user enters in the details
of the area they wish the generate terrain for, then the website will generate a terrain.zip file containing
the relevant dat files. The user will download this file and then
then need to unzip to a "terrain" folder on the SD card in their flight controller.

## Pre-generation of Terrain

To ensure the website operates responsively, the terrain for the whole (-60 -> +60 latitude for SRTM3, 
-84 -> +84 latitude for SRTM1) world
must be pregenerated. This will take some time.

Run ``offline_gen.py`` to download the SRTM files from ardupilot.org and convert them to the dat
file format. These files will be stored in the processedTerrain folder.

## For developers

This website uses the flask library.

To install dependencies:

``pip install -r requirements.txt``

To run:

```
python3 app.py
```

The unzipped processed files are temporarily stored in ./outputTer-tmp. These are deleted upon the zipping into a single
downloadable file

The downloadable files are stored in ./outputTer

Each user request is given a UUID, which is incorporated into the folder/filename of the terrain files.

To run the unit tests, type ``pytest``

A systemd service is provided for running the WSGI server.

## Tools

- **fast_gen.py** - Fast terrain DAT file generator using numpy. Generates `.DAT.gz` files from SRTM HGT data with multiprocessing. Supports both land and ocean tiles, and both SRTM1 (30m) and SRTM3 (100m) spacing. Use `--lat-range` to generate ocean tiles for a latitude range.

- **create_filelist.py** - Creates the `filelist_python` pickle file for an HGT directory. This file is used by `srtm.py` to look up available tiles without scanning the directory each time.

- **create_continents.py** - Packages terrain DAT files into per-continent ZIP archives for distribution, using the filelist_python continent mapping from `srtm.py`.

- **resample_hgt.py** - Downsamples SRTM1 HGT files (3601x3601) to SRTM3 (1201x1201) by taking every 3rd pixel. Takes input and output directory names.

- **srtm1_to_srtm3.py** - Resamples SRTM1 HGT data to SRTM3 resolution using mean pooling via scipy interpolation.

- **find_steep.py** - Scans HGT or DAT files for neighbouring points exceeding a height difference threshold. Reports count, max difference, and lat/lon of the steepest point. Useful for finding data anomalies.

- **slice_graph.py** - Altitude profile visualiser. Plots a horizontal slice through terrain tiles comparing DAT and HGT data side by side, replicating both AP_Terrain and srtm.py interpolation methods.

- **terrain_view.py** - 2D terrain visualiser. Displays DAT or HGT files as colour-mapped images with mouse-over lat/lon and height readout. Supports `--diff` mode to compare two files.

- **terrain_gen.py** - (Deprecated, use fast_gen.py) Original terrain DAT file generator. Used by offline_gen.py and app.py for core data structures and coordinate calculations.

- **version_minor.py** - Reads or sets the `version_minor` field in terrain `.DAT.gz` files. Used to mark regenerated tiles so ArduPilot can detect outdated terrain data.

- **offline_check.py** - Validates terrain DAT files for corruption by checking CRC, block structure, and coverage.

