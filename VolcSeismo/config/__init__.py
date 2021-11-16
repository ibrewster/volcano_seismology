import configparser
import os


def load_config():
    config = configparser.ConfigParser()
    script_loc = os.path.dirname(__file__)
    conf_file = os.path.join(script_loc, 'config.ini')
    try:
        config.read(conf_file)
    except FileNotFoundError:
        pass
    return config


config = load_config()

try:
    from .location_config import Locations, Stations
except ImportError:
    locations = {}
    stations = {}
else:
    # instantiate an instance of the locations class
    locations = Locations()
    stations = Stations()

