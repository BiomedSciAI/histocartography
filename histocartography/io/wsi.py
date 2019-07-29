"""Whole Slide Image IO module."""
import logging
import sys
import pys3

# setup logging
#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
log = logging.getLogger('Histocartography::IO::WSI')
h1 = logging.StreamHandler(sys.stdout)
log.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
h1.setFormatter(formatter)
log.addHandler(h1)


def load(file=None):
    """TODO. Currently returns method name"""
    log.info("Loading file: {}".format(file))
    return 'Load'


def save(file=None):
    """TODO. Currently returns method name"""
    log.info("Saving file: {}".format(file))
    return 'Save'

def patch(file=None):
    """TODO. Currently returns method name"""
    log.info("Getting patch from file: {}".format(file))
    return 'Patch'

