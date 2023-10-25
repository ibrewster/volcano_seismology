import os

import sys
sys.path.append('..')

import numpy

from obspy import UTCDateTime

from .dVv.compile_msnoise_B2 import main
from VolcSeismo.waveform import load

def run(stream, times, station, metadata):
    # save stream to a file in the desired location
    # run main passing said location as an argument

    STA = stream[0].stats.station
    NET = metadata['NET']

    end_time = times.max()
    start_time = end_time - (24 * 60 * 60 * 1000)

    start_str = str(numpy.datetime_as_string(start_time, unit = 'D'))
    end_str = str(numpy.datetime_as_string(end_time, unit = 'D'))

    data_location = os.path.join(os.path.dirname(__file__), 'dVv', 'processing', station, 'data')
    datafile_loc = os.path.join(data_location, NET, STA)
    filename = f"{times[0]}-{times[-1]}.mseed"
    filename = os.path.join(datafile_loc, filename)
    output_dir = os.path.abspath(os.path.join(data_location, '..', 'output'))

    os.makedirs(output_dir, exist_ok = True)
    os.makedirs(datafile_loc, exist_ok = True)

    # Convert to int32 for compatibility with the MSEED output format
    for tr in stream:
        tr.data = tr.data.astype(numpy.int32)

    stream.write(filename, format = "MSEED")

    # Make sure we have half an hour of data (at least)
    check_start = end_time - (60 * 40 * 1000)

    CHAN = metadata['CHAN']
    START = check_start - (60 * 10 * 1000)
    while START < end_time:
        START += (60 * 10 * 1000) # 10 minutes
        END = START + (60 * 10 * 1000)
        STARTTIME = UTCDateTime(str(START))
        ENDTIME = UTCDateTime(str(END))
        filename_start = START.astype('datetime64[m]').astype('datetime64[ms]')
        fill_filename =f"{filename_start}-{END}.mseed"
        fill_filepath = os.path.join(datafile_loc, fill_filename)
        if not os.path.isfile(fill_filepath):
            stream, waveform_times = load(NET, STA, '--', CHAN, STARTTIME, ENDTIME)
            for tr in stream:
                tr.data = tr.data.astype(numpy.int32)

            stream.write(fill_filepath, format = "MSEED")

    main(data_location, start_str, end_str)
