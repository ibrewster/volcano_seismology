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
           if isfile(f)
           and not f.endswith('__init__.py')
           and not f.startswith("_")]


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
        z_stream = stream.select(component = 'Z').pop()
        z_channel = z_stream.stats['channel']
        z_data = z_stream.data
    except:
        z_data = []
        z_channel = None

    try:
        n_stream = stream.select(component = 'N').pop()
        n_channel = n_stream.stats['channel']
        n_data = n_stream.data
    except Exception as e:
        n_data = []
        n_channel = None

    try:
        e_stream = stream.select(component = 'E').pop()
        e_data = e_stream.data
        e_channel = e_stream.stats['channel']
    except Exception as e:
        e_data = []
        e_channel = None

    data_df = _create_df(times, z_data, n_data, e_data)
    metadata = {'Z': z_channel,
                'N': n_channel,
                'E': e_channel, }
    station = stream.traces[0].stats['station']
    for hook in __all__:
        try:
            globals()[hook].run(data_df, station, metadata)
        except AttributeError:
            pass  # We already warned this hook would be unavailable
        except TypeError as e:
            warnings.warn(f"Unable to run hook '{hook}' {e}",
                          HookWarning,
                          stacklevel=2)
            pass  # No run function, or bad signature
