from importlib.metadata import version

from pyxeol import laue
from pyxeol import misc
from pyxeol import readMDA
from pyxeol import scanInfo
from pyxeol import specfun
from pyxeol import xbic
from pyxeol import xeol
from pyxeol import xrf


try:
    __version__ = version(__package__)
except:
    __version__ = 'testing'
