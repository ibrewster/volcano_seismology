import logging
import multiprocessing
import os
import re
import time

from concurrent.futures import ProcessPoolExecutor

import pandas
import psycopg

from dVv.compile_msnoise_B2 import main

from obspy import UTCDateTime

try:
    from dVv import wingdbstub
except ImportError:
    print("Unable to import WingDBStub. Not debugging.")


NET_ID = re.compile('A[VK]\.')

class DBCursor():
    def __init__(self, cursor_factory=None):
        self._cursor_factory = cursor_factory

    def __enter__(self):
        self._conn = psycopg.connect(host='137.229.113.120',
                                     dbname='volcano_seismology',
                                     cursor_factory=self._cursor_factory,
                                     user="geodesy",
                                     password = 'G30dE$yU@F')
        self._cursor = self._conn.cursor()
        return self._cursor

    def __exit__(self, *args, **kwargs):
        try:
            self._conn.rollback()
        except AttributeError:
            return  # No connection
        self._conn.close()

def get_lookups():
    # Get a list of stations and associated volcanoes
    SQL = """
    SELECT name,id
    FROM stations
    """

    with DBCursor() as cursor:
        cursor.execute("SELECT name,id FROM stations")
        station_lookup = {x[0]: int(x[1]) for x in cursor}

        cursor.execute("SELECT site,id FROM volcanoes")
        volc_lookup = {x[0]: x[1] for x in cursor}

    return (station_lookup, volc_lookup)


STA_LOOKUP, VOLC_LOOKUP = None, None

def init_lookups():
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        print("Context already set", multiprocessing.get_start_method())
    global STA_LOOKUP
    global VOLC_LOOKUP

    STA_LOOKUP, VOLC_LOOKUP = get_lookups()

def process_dVv(data_location, output_dir, start_str, end_str):
    main(data_location, output_dir, start_str, end_str)
    tt_file = os.path.join(output_dir, 'tt.csv')
    if not os.path.exists(tt_file):
        values = []
    else:
        data = pandas.read_csv(tt_file, parse_dates = ['Date'])
        pairs = data['Pairs'].str.replace(NET_ID, '', regex = True)
        pairs = pairs.replace(['ALL'], [None])
        pairs = pairs.str.split('._', expand = True)
        pairs[1] = pairs[1].str.replace('.', '', regex = False)
        del data['Pairs']
        data['sta1'] = pairs[0].replace(STA_LOOKUP)
        data['sta2'] = pairs[1].replace(STA_LOOKUP)
        values = data.to_dict(orient = 'records')

    return values


if __name__ == "__main__":
    init_lookups()

    t1 = time.time()
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'dVv', 'processing'))
    volcs = [x for x in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, x))]

    start_date = UTCDateTime.now()
    end_date = UTCDateTime.now()

    # Just process one day
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    # DEBUG
    # start_str, end_str = '2023-09-30', '2023-10-01'
    procs = []
    all_values = []
    with ProcessPoolExecutor(initializer = init_lookups) as executor:
        for volc in volcs:
            if volc == 'Unknown':
                continue

            data_location = os.path.abspath(os.path.join(data_path, volc, 'data'))
            output_dir = os.path.abspath(os.path.join(data_location, '..', 'Output'))
            try:
                os.unlink(os.path.join(output_dir, 'tt.csv'))
            except FileNotFoundError:
                pass

            proc = executor.submit(process_dVv, data_location, output_dir, start_str, end_str)
            # ret = process_dVv(data_location, output_dir, start_str, end_str)
            # proc = FakeProc(ret)
            procs.append((volc, proc))

    for volc, proc in procs:
        try:
            results = proc.result()
            if not results:
                print(f"********No results generated for {volc}")
            else:
                for x in results:
                    x['volc'] = VOLC_LOOKUP[volc]

                all_values += results
        except Exception as e:
            logging.exception("Unable to generate results for %s: %s",
                              volc, str(e))

    value_sql ="(%(Date{idx})s, $(volc{idx})s, $(sta1{idx})s, %(sta2{idx})s, %(M{idx})s, %(EM{idx})s, %(A{idx})s, %(EA{idx})s, %(M0{idx})s, %(EM0{idx})s)"

    sql_placeholders = []
    args = {}
    for idx, value_dict in enumerate(all_values):
        sql_placeholders.append(value_sql.format(idx = idx))
        for key, value in value_dict:
            key += str(idx)
            args[key] = value


    SQL = "INSERT INTO dvv (datetime,volc,sta1,sta2,m,em,a,ea,m0,em0) VALUES\n"
    SQL += ",\n".join(sql_placeholders)
    SQL += """ON CONFLICT (datetime,sta1,sta2) DO UPDATE SET
    m=EXCLUDED.m,
    em=EXCLUDED.em,
    a=EXCLUDED.a,
    ea=EXCLUDED.ea,
    m0=EXCLUDED.m0,
    em0=EXCLUDED.em0"""

    with DBCursor() as cursor:
        cursor.execute(SQL, args)
        cursor.connection.commit()

    print("***Complete in", (time.time() - t1) / 60)
