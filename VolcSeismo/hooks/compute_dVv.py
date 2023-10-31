import os

import sys

import obspy
import pandas

sys.path.append('..')

import numpy

from obspy import UTCDateTime

from .dVv.compile_msnoise_B2 import main
from VolcSeismo.waveform import load
from VolcSeismo.web import utils


volc_lookup = {}
def init():
    global volc_lookup
    # Get a list of stations and associated volcanoes
    SQL = """
    SELECT DISTINCT ON (name)
        name,
        volcanoes.site
    FROM stations
    CROSS JOIN volcanoes
    WHERE stations.location <-> volcanoes.location < (volcanoes.radius*1000)
    ORDER BY name, stations.location <-> volcanoes.location
    """

    with utils.db_cursor() as cursor:
        cursor.execute(SQL)
        volc_lookup = {x[0]: x[1] for x in cursor}

init()

def run(stream, times, station, metadata):
    # save stream to a file in the desired location
    # run main passing said location as an argument

    STA = stream[0].stats.station
    NET = metadata['NET']
    CHAN = metadata['CHAN']
    VOLC = volc_lookup.get(station, 'Unknown')

    end_time = times.max()
    data_start = pandas.to_datetime(times.min())
    start_time = end_time - (24 * 60 * 60 * 1000)

    start_str = str(numpy.datetime_as_string(start_time, unit = 'D'))
    end_str = str(numpy.datetime_as_string(end_time, unit = 'D'))

    data_location = os.path.join(os.path.dirname(__file__), 'dVv', 'processing', VOLC, 'data')
    datafile_loc = os.path.join(data_location, NET, STA)

    data_year = data_start.year
    data_doy = data_start.dayofyear

    filename = f"{STA}.{NET}...{data_year}.{data_doy}"
    file = os.path.join(datafile_loc, filename)

    # Convert to int32 for compatibility with the MSEED output format
    for tr in stream:
        tr.data = tr.data.astype(numpy.int32)


    target_start = stream[0].times('utcdatetime').max() - (60 * 40)

    try:
        data = obspy.read(file)
        data += stream
    except FileNotFoundError:
        data = stream

    data_start = data[0].times('utcdatetime').min()

    output_dir = os.path.abspath(os.path.join(data_location, '..', 'Output'))

    os.makedirs(output_dir, exist_ok = True)
    os.makedirs(datafile_loc, exist_ok = True)

    # Make sure we have the full half hour of data
    if data_start > target_start:
        stream, waveform_times = load(NET, STA, '--', CHAN, target_start, data_start)

        for tr in stream:
            tr.data = tr.data.astype(numpy.int32)

        data += stream

    data = data.merge(method = 1, fill_value = 'latest',
                      interpolation_samples = -1)

    data = data.trim(target_start, nearest_sample = False)

    data_start = data[0].times('utcdatetime').min()
    data_end = data[0].times('utcdatetime').max()

    if data_start.day != data_end.day:
        slice_point = data_end.replace(hour = 0, minute = 0, second = 0, microsecond = 0)
        first_data = data.slice(data_start, slice_point-1)
        data = data.slice(slice_point, data_end)

        first_data_doy = data_start.julday
        first_data_year = data_start.year
        first_filename = f"{STA}.{NET}...{first_data_year}.{first_data_doy}"
        first_file = os.path.join(datafile_loc, first_filename)
        first_data.write(first_file, format = "MSEED")


    filename = f"{STA}.{NET}...{data_year}.{data_doy}"
    file = os.path.join(datafile_loc, filename)
    data.write(file, format = "MSEED")

    #start_str, end_str ='2021-08-05', '2021-08-07'
    #main(data_location, output_dir, start_str, end_str)
