from setuptools import setup

APP = ['mac_audiogen.py']
#OPTIONS = { 'includes': ['EXTERNAL LIBRARY'],}

DATA_FILES = []
OPTIONS = {
    'argv_emulation': True,
    'iconfile':'icon.ico'
}

setup(
    app=APP,
    name='AudioGen',
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
