import io
import os
import time
import pytest
import struct
import zipfile

from app import app

def createFile(name, size):
    # create a file of random data
    with open(name, 'wb') as fout:
        fout.write(os.urandom(size)) # replace 1024 with size_kb if not unreasonably large


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_homepage(client):
    """Test that the homepage can be generated"""

    rv = client.get('/')
    # assert all the controls are there
    assert b'<title>ArduPilot Terrain Generator</title>' in rv.data
    assert b'<form action="/generate" method="post">' in rv.data
    assert b'<input type="text" id="lat" name="lat" value="-35.363261" oninput="plotCircleCoords(true);">' in rv.data
    assert b'<input type="text" id="long" name="long" value="149.165230" oninput="plotCircleCoords(true);">' in rv.data
    assert b'<input type="range" id="radius" name="radius" value="100" min="1" max="400"\n                oninput="plotCircleCoords(false);">' in rv.data
    assert b'<input type="submit" value="Generate" method="post">' in rv.data
    assert b'<select name="version" id="version">' in rv.data

def test_badinput(client):
    """Test bad inputs"""
    # no input
    rv = client.post('/generate', data=dict(
    ), follow_redirects=True)

    assert b'<title>ArduPilot Terrain Generator</title>' in rv.data
    assert b'Error' in rv.data
    assert b'Link To Download' not in rv.data

    #partial input
    rv = client.post('/generate', data=dict(
        lat='-35.363261',
        long='149.165230',
        version="1"
    ), follow_redirects=True)

    assert b'<title>ArduPilot Terrain Generator</title>' in rv.data
    assert b'Error' in rv.data
    assert b'Link To Download' not in rv.data

    #bad lon/lat
    rv = client.post('/generate', data=dict(
        lat='I am bad data',
        long='echo test',
        radius='1',
        version="1"
    ), follow_redirects=True)

    assert b'<title>ArduPilot Terrain Generator</title>' in rv.data
    assert b'Error' in rv.data
    assert b'download="terrain.zip"' not in rv.data

    #out of bounds lon/lat
    rv = client.post('/generate', data=dict(
        lat='206.56',
        long='-400',
        radius='1',
        version="3"
    ), follow_redirects=True)

    assert b'<title>ArduPilot Terrain Generator</title>' in rv.data
    assert b'Error' in rv.data
    assert b'download="terrain.zip"' not in rv.data
    
def test_simplegen_1(client):
    """Test that a small piece of terrain can be generated, SRTM1"""

    rv = client.post('/generate', data=dict(
        lat='60.363261',
        long='167.165230',
        radius='1',
        version="1"
    ), follow_redirects=True)

    assert b'<title>ArduPilot Terrain Generator</title>' in rv.data
    assert b'Error' not in rv.data
    assert b'Tiles were requested which are not covered by the terrain database' not in rv.data
    assert b'download="terrain.zip"' in rv.data

    uuidkey = (rv.data.split(b"footer")[1][1:-2]).decode("utf-8")
    assert uuidkey != ""

    #file should be ready for download and around 2MB in size
    rdown = client.get('/userRequestTerrain/' + uuidkey + ".zip", follow_redirects=True)
    assert b'404 Not Found' not in rdown.data
    assert len(rdown.data) > (1*1024*1024)

    # check file has the appropriate data files inside it
    file_like_object = io.BytesIO(rdown.data)
    with zipfile.ZipFile(file_like_object) as zip_file:
        file_list = zip_file.namelist()
        assert file_list == ['N60E167.DAT']

def test_simplegen_3(client):
    """Test that a small piece of terrain can be generated, SRTM3"""

    rv = client.post('/generate', data=dict(
        lat='-35.363261',
        long='149.165230',
        radius='1',
        version="3"
    ), follow_redirects=True)

    assert b'<title>ArduPilot Terrain Generator</title>' in rv.data
    assert b'Error' not in rv.data
    assert b'Tiles were requested which are not covered by the terrain database' not in rv.data
    assert b'download="terrain.zip"' in rv.data

    uuidkey = (rv.data.split(b"footer")[1][1:-2]).decode("utf-8")
    assert uuidkey != ""

    #file should be ready for download and around 2MB in size
    rdown = client.get('/userRequestTerrain/' + uuidkey + ".zip", follow_redirects=True)
    assert b'404 Not Found' not in rdown.data
    assert len(rdown.data) > (1*1024*1024)

    # check file has the appropriate data files inside it
    file_like_object = io.BytesIO(rdown.data)
    with zipfile.ZipFile(file_like_object) as zip_file:
        file_list = zip_file.namelist()
        assert file_list == ['S36E149.DAT']
    
def test_simplegenoutside(client):
    """Test that a small piece of terrain can be generated with partial outside +-84latitude for SRTM3"""

    rv = client.post('/generate', data=dict(
        lat='-83.363261',
        long='149.165230',
        radius='100',
        version="3"
    ), follow_redirects=True)

    assert b'<title>ArduPilot Terrain Generator</title>' in rv.data
    assert b'Error' not in rv.data
    assert b'Tiles were requested which are not covered by the terrain database' in rv.data
    assert b'download="terrain.zip"' in rv.data

    uuidkey = (rv.data.split(b"footer")[1][1:-2]).decode("utf-8")
    assert uuidkey != ""

    #file should be ready for download and around 2MB in size
    rdown = client.get('/userRequestTerrain/' + uuidkey + ".zip", follow_redirects=True)
    assert b'404 Not Found' not in rdown.data
    assert len(rdown.data) > (0.25*1024*1024)

    # check file has the appropriate data files inside it
    file_like_object = io.BytesIO(rdown.data)
    with zipfile.ZipFile(file_like_object) as zip_file:
        file_list = zip_file.namelist()
        assert len(file_list) == 33
        assert set(file_list) == set(['S84E141.DAT', 'S83E141.DAT',
                                      'S84E142.DAT', 'S83E142.DAT',
                                      'S84E143.DAT', 'S83E143.DAT',
                                      'S84E144.DAT', 'S83E144.DAT',
                                      'S84E145.DAT', 'S83E145.DAT',
                                      'S84E146.DAT', 'S83E146.DAT',
                                      'S84E147.DAT', 'S83E147.DAT',
                                      'S84E148.DAT', 'S83E148.DAT',
                                      'S84E149.DAT', 'S83E149.DAT',
                                      'S84E150.DAT', 'S83E150.DAT',
                                      'S84E151.DAT', 'S83E151.DAT',
                                      'S84E152.DAT', 'S83E152.DAT',
                                      'S84E153.DAT', 'S83E153.DAT',
                                      'S84E154.DAT', 'S83E154.DAT',
                                      'S84E155.DAT', 'S83E155.DAT',
                                      'S84E156.DAT', 'S83E156.DAT',
                                      'S84E157.DAT'])

def test_multigen(client):
    """Test that a a few small piece of terrains can be generated"""

    rva = client.post('/generate', data=dict(
        lat='-35.363261',
        long='149.165230',
        radius='1',
        version="3"
    ), follow_redirects=True)
    time.sleep(0.1)

    rvb = client.post('/generate', data=dict(
        lat='-35.363261',
        long='147.165230',
        radius='10',
        version="1"
    ), follow_redirects=True)
    time.sleep(0.1)

    rvc = client.post('/generate', data=dict(
        lat='-30.363261',
        long='137.165230',
        radius='100',
        version="3"
    ), follow_redirects=True)
    time.sleep(0.1)

    # Assert reponse is OK and get UUID for each ter gen
    allUuid = []
    for rv in [rva, rvb, rvc]:
        assert b'<title>ArduPilot Terrain Generator</title>' in rv.data
        assert b'Error' not in rv.data
        assert b'download="terrain.zip"' in rv.data
        uuidkey = (rv.data.split(b"footer")[1][1:-2]).decode("utf-8")
        assert uuidkey != ""
        allUuid.append(uuidkey)

    #files should be ready for download and around 0.7MB in size
    for uukey in allUuid:
        rdown = client.get('/userRequestTerrain/' + uukey + ".zip", follow_redirects=True)
        assert b'404 Not Found' not in rdown.data
        assert len(rdown.data) > (0.7*1024*1024)

def test_coastal_nonzero_altitude(client):
    """Test that coastal regions have correct non-zero altitudes.

    This tests for a bug where land tiles incorrectly got zero altitude
    when processed after ocean tiles due to using a stale tile variable.
    Location: New Zealand coast at -44.5234, 171.00197
    """
    rv = client.post('/generate', data=dict(
        lat='-44.5234',
        long='171.00197',
        radius='1',
        version="3"
    ), follow_redirects=True)

    assert b'<title>ArduPilot Terrain Generator</title>' in rv.data
    assert b'Error' not in rv.data
    assert b'download="terrain.zip"' in rv.data

    uuidkey = (rv.data.split(b"footer")[1][1:-2]).decode("utf-8")
    assert uuidkey != ""

    # Download the generated terrain
    rdown = client.get('/userRequestTerrain/' + uuidkey + ".zip", follow_redirects=True)
    assert b'404 Not Found' not in rdown.data

    # Extract and check the DAT file
    file_like_object = io.BytesIO(rdown.data)
    with zipfile.ZipFile(file_like_object) as zip_file:
        file_list = zip_file.namelist()
        assert 'S45E171.DAT' in file_list

        # Read the DAT file and check for non-zero altitudes
        dat_data = zip_file.read('S45E171.DAT')

        # DAT file structure: 2048 byte blocks
        # Each block has: bitmap(8) + lat(4) + lon(4) + crc(2) + version(2) + spacing(2) = 22 bytes header
        # Then 32*28 = 896 heights as signed 16-bit integers
        IO_BLOCK_SIZE = 2048
        num_blocks = len(dat_data) // IO_BLOCK_SIZE

        max_altitude = 0
        nonzero_count = 0
        total_heights = 0

        for block_idx in range(num_blocks):
            block_start = block_idx * IO_BLOCK_SIZE
            block = dat_data[block_start:block_start + IO_BLOCK_SIZE]

            # Skip header (22 bytes), read heights
            height_data = block[22:22 + 896*2]  # 896 heights * 2 bytes each
            heights = struct.unpack('<%dh' % 896, height_data)

            for h in heights:
                total_heights += 1
                if h > 0:
                    nonzero_count += 1
                    max_altitude = max(max_altitude, h)

        # This location is on land in New Zealand - should have significant non-zero altitudes
        # The actual altitude at -44.5234, 171.00197 is around 180m
        assert max_altitude > 100, f"Expected max altitude > 100m for NZ land, got {max_altitude}m"
        assert nonzero_count > 0, "Expected some non-zero altitudes for land area"

def test_terrain_gen_coastal_bug():
    """Test that terrain_gen.create_degree produces non-zero altitudes for coastal land.

    This directly tests the fix for a bug where land tiles incorrectly got zero
    altitude when processed after ocean tiles due to using a stale tile variable.
    """
    import tempfile
    import shutil
    from srtm import SRTMDownloader
    from terrain_gen import create_degree

    # Set up downloader for SRTM3
    downloader = SRTMDownloader(debug=False, offline=0, directory="SRTM3")
    downloader.loadFileList()

    # Create temp folder for output
    temp_dir = tempfile.mkdtemp()
    try:
        # S45E171 - New Zealand coast, lat=-45, lon=171
        lat = -45
        lon = 171
        spacing = 100  # SRTM3
        format = "4.1"

        # Wait for download and generate tile
        max_wait = 120
        start = time.time()
        result = False
        while time.time() - start < max_wait:
            result = create_degree(downloader, lat, lon, temp_dir, spacing, format)
            if result:
                break
            time.sleep(1)

        assert result, "Failed to generate terrain tile"

        # Read and check the generated DAT file
        dat_file = os.path.join(temp_dir, "S45E171.DAT")
        assert os.path.exists(dat_file), "DAT file not created"

        with open(dat_file, 'rb') as f:
            dat_data = f.read()

        IO_BLOCK_SIZE = 2048
        num_blocks = len(dat_data) // IO_BLOCK_SIZE

        max_altitude = 0
        nonzero_count = 0

        for block_idx in range(num_blocks):
            block_start = block_idx * IO_BLOCK_SIZE
            block = dat_data[block_start:block_start + IO_BLOCK_SIZE]

            # Skip header (22 bytes), read heights
            height_data = block[22:22 + 896*2]
            heights = struct.unpack('<%dh' % 896, height_data)

            for h in heights:
                if h > 0:
                    nonzero_count += 1
                    max_altitude = max(max_altitude, h)

        # This tile covers NZ Southern Alps - should have high peaks
        assert max_altitude > 1000, f"Expected max altitude > 1000m for NZ Alps, got {max_altitude}m"
        assert nonzero_count > 100000, f"Expected many non-zero altitudes, got {nonzero_count}"

    finally:
        shutil.rmtree(temp_dir)
