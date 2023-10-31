import os
from dVv.compile_msnoise_B2 import main

from obspy import UTCDateTime



if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), 'dVv', 'processing')
    volcs = os.listdir(data_path)

    end_date = UTCDateTime.now()
    start_date = end_date - (60 * 60 * 24) #One day earlier

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    # DEBUG
    start_str, end_str = '2023-09-30', '2023-10-01'
    volcs = ['Okmok']
    for volc in volcs:
        data_location = os.path.join(data_path, volc, 'data')
        output_dir = os.path.abspath(os.path.join(data_location, '..', 'Output'))

        main(data_location, output_dir, start_str, end_str)
