import os
import time
import pytest

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
    
def test_simplegenoutside(client):
    """Test that a small piece of terrain can be generated with partial outside +-84latitude for SRTM3"""

    rv = client.post('/generate', data=dict(
        lat='-83.363261',
        long='149.165230',
        radius='200',
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
