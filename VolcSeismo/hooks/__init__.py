import glob
import importlib
import itertools
import warnings

from os.path import dirname, basename, isfile, join

import numpy
import pandas

from obspy import UTCDateTime

modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [basename(f)[:-3] for f in modules
           if isfile(f) and
           not f.endswith('__init__.py') and
           not f.startswith("_")]


class HookWarning(UserWarning):
    pass


for module in __all__:
    try:
        importlib.import_module("." + module, package = __name__)
    except ImportError as e:
        warnings.warn(f"Unable to import hook {module}. This hook will not be available.\n{e}",
                      HookWarning, 2)


def _create_df(times, z_data, n_data, e_data):
    if not numpy.asarray([
        numpy.asarray(z_data).size,
        numpy.asarray(n_data).size,
        numpy.asarray(e_data).size
    ]).any():
        raise TypeError("Need at least one of Z, N, or E channel data")

    data = itertools.zip_longest(times, z_data, n_data, e_data,
                                 fillvalue = numpy.nan)
    headers = ['time', 'Z', 'N', 'E']
    df = pandas.DataFrame(data = data, columns = headers)
    return df


def run_hooks(stream, times = None):
    if not __all__:
        return

    if times is None:
        times = stream[0].times()
        DATA_START = UTCDateTime(stream[0].stats['starttime'])
        times = ((times + DATA_START.timestamp) * 1000).astype('datetime64[ms]')

    try:
        z_data = stream.select(component = 'Z').pop().data
    except:
        z_data = []

    try:
        n_data = stream.select(component = 'N').pop().data
    except Exception as e:
        n_data = []

    try:
        e_data = stream.select(component = 'E').pop().data
    except Exception as e:
        e_data = []

    data_df = _create_df(times, z_data, n_data, e_data)
    station = stream.traces[0].get_id().split('.')[1]
    for hook in __all__:
        try:
            globals()[hook].run(data_df, station)
        except AttributeError:
            pass  # We already warned this hook would be unavailable
        except TypeError as e:
            warnings.warn(f"Unable to run hook '{hook}' {e}",
                          HookWarning,
                          stacklevel=2)
            pass  # No run function, or bad signature
