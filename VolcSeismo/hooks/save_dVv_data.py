import os

import sys

import obspy

sys.path.append('..')

import numpy

from obspy import UTCDateTime

from VolcSeismo.waveform import load
from VolcSeismo.web import utils
from VolcSeismo import config


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

    data_location = os.path.join(os.path.dirname(__file__), 'dVv', 'processing', VOLC, 'data')
    datafile_loc = os.path.join(data_location, NET, STA)
    output_dir = os.path.abspath(os.path.join(data_location, '..', 'Output'))

    os.makedirs(output_dir, exist_ok = True)
    os.makedirs(datafile_loc, exist_ok = True)

    # Convert to int32 for compatibility with the MSEED output format
    for tr in stream:
        tr.data = tr.data.astype(float)

    file = os.path.join(datafile_loc, "data.mseed")

    try:
        data = obspy.read(file)
        data += stream
    except FileNotFoundError:
        data = stream

    # Find the last time in this stream. Could be any of the traces, so...
    data_end = UTCDateTime(0)
    for tr in stream:
        data_times = tr.times('timestamp')
        data_end = max(data_end, UTCDateTime(data_times.max()))

    target_start = UTCDateTime(data_times.max()) - (60 * 60)
    target_start = target_start.replace(second = 0, microsecond = 0)

    # Find closest half-hour mark
    approx = round(target_start.minute/30) *30
    target_start = target_start.replace(minute = 0) + (60 * approx)

    # Discard any old data, so we know if we have current or not.
    data = data.trim(target_start)

    # Now that we loaded the existing data, see how far back it *actually* goes
    # Theoretically this *should* be target start now, since we just trimmed, but
    # if the data hasn't been updated in a while, it may be different.
    data_times = data[0].times('timestamp') # Don't get UTCDateTimes directly, as that is WAY slow.
    data_start = UTCDateTime(data_times.min())

    # Make sure we have the full amount of data
    if data_start > target_start:
        stream, waveform_times = load(NET, STA, '--', CHAN, target_start, data_start, config.availability)

        # If we managed to get more data, append it to the data file.
        if stream:
            for tr in stream:
                tr.data = tr.data.astype(float)

            data += stream

    data = data.merge(method = 1, fill_value = 'latest',
                      interpolation_samples = -1)

    # Make sure we are exactly on the start mark.
    data = data.trim(target_start, nearest_sample = False)

    # Replace any NaN values with zero
    for tr in data:
        tr.interpolate(method='zero', sampling_rate=tr.meta['sampling_rate'])


    # And write out the new data file
    data.write(file, format = "MSEED")
