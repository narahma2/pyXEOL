from importlib.metadata import version

from pyxeol import laue
from pyxeol import misc
from pyxeol import specfun
from pyxeol import xeol


try:
    __version__ = version(__package__)
except:
    __version__ = 'testing'
