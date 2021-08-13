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
    from .station_config import locations, stations
except ImportError:
    locations = {}
    stations = {}
