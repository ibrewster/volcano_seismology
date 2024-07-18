import json
import logging
import multiprocessing
import os
import re
import sys
import time

from concurrent.futures import ProcessPoolExecutor

import numpy
import pandas
import psycopg

from pathlib import Path

from dVv.compile_msnoise_C2 import main as run_msnoise
from dVv.CleanCSV import clean_output

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
    with DBCursor() as cursor:
        cursor.execute("SELECT name,id FROM stations")
        station_lookup = {x[0]: int(x[1]) for x in cursor}
        station_lookup[numpy.nan] = station_lookup['__AVG']
        station_lookup[None] = station_lookup['__AVG']

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

def process_dVv(data_location, output_dir, start_str, end_str, volc):
    try:
        run_msnoise(data_location, output_dir, start_str, end_str)
    except Exception as e:
        raise ValueError(e)
    
    return process_output(output_dir, volc)

def process_output(output_dir, volc):
    """Three files: dvv, coh, and error."""
    global STA_LOOKUP
    global VOLC_LOOKUP
    if STA_LOOKUP is None:
        stas, volcs = get_lookups()
        STA_LOOKUP = stas
        VOLC_LOOKUP = volcs

    pair_base: Path = Path(output_dir) / "WCT" / "01" / "001_DAYS" / "ZZ"

    output = pandas.DataFrame(columns = ['sta1', 'sta2', 'datetime', 'dvv', 'coh', 'err'])
    output = output.set_index(['sta1', 'sta2', 'datetime'])
    
    print(f"--------Compiling result data--------------")

    for pair_dir in pair_base.iterdir():
        if not pair_dir.is_dir():
            continue  # Probably something like .DS_Store

        parts = pair_dir.name.split('_')
        sta1 = STA_LOOKUP[parts[1]]
        sta2 = STA_LOOKUP[parts[3]]
        
        print(f"----------{parts[1]}--{parts[3]}------------")

        clean_output(pair_dir, sta1, sta2)
        
        for csv_file in pair_dir.glob("*.csv"):
            data = pandas.read_csv(csv_file, parse_dates = [0])
            data['sta1'] = sta1
            data['sta2'] = sta2
            data = data.set_index(['sta1', 'sta2', 'Unnamed: 0'])
            data = data.replace([numpy.nan], [None], regex=False).to_dict(orient = 'index')

            col = 'dvv' # default/else
            if "coh" in csv_file.name:
                col = 'coh'
            elif "error" in csv_file.name:
                col = 'err'

            for key, value in data.items():
                output.at[key, col] = json.dumps(value)

    # "flatten" index into records
    output = output.reset_index()
    output['volc'] = volc
    output['datetime'] = pandas.Series(output['datetime'].dt.to_pydatetime(), dtype = object)
    # values = output.to_dict(orient = "records")
    return output


def run_compute():
    os.environ['PATH'] = os.path.dirname(sys.executable) +  os.pathsep + os.environ['PATH']
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
    workers = min(len(volcs), 15)
    print(f"--------Using {workers} workers-------")
    with ProcessPoolExecutor(initializer=init_lookups, max_workers=workers) as executor:
        for volc in volcs:
            if volc != 'Veniaminof':
                continue
            
            if volc == 'Unknown':
                continue

            print(f"--------------Submitting {volc} for processing-----------------")

            data_location = os.path.abspath(os.path.join(data_path, volc, 'data'))
            output_dir = os.path.abspath(os.path.join(data_location, '..', 'Output'))

            for old_file in ['tt.csv', 'db.ini', 'msnoise.sqlite']:
                try:
                    os.unlink(os.path.join(output_dir, old_file))
                except FileNotFoundError:
                    pass

            proc = executor.submit(process_dVv, data_location, output_dir, start_str, end_str, VOLC_LOOKUP[volc])
            procs.append((volc, proc))
            #############DEBUG##################
            # try:                
                # result = process_dVv(
                    # data_location, output_dir, start_str, end_str, VOLC_LOOKUP[volc]
                # )
                # procs.append((volc, result))
            # except ValueError as e:
                # print(f"Unable to process volcano {volc}. Error: {e}")
            ####################################

        print("--------------Jobs Submitted. Waiting for results---------------")
        for volc, proc in procs:
            print(f"--------Processing result for {volc}--------------")
            try:
                results = proc.result()
                ########## DEBUG ###########
                # results = proc
                ############################
                if len(results) == 0:
                    print(f"********No results generated for {volc}*************")
                else:
                    print(f"--------Submitting results for {volc} to db--------------")
                    submit_results(results)
            except Exception as e:
                logging.exception(
                    "!!!!!!!!!!!!!!Unable to generate results for %s: %s!!!!!!!!!!!",
                    volc,
                    str(e),
                )
    
        print("***Complete in", (time.time() - t1) / 60)


def submit_results(value_df):
    #  Keep this value small enough to not overload the system
    chunk_size = 1000
    print("Total length:", len(value_df), "Processing in chunks of:", chunk_size)
    for idx, value_chunk in value_df.groupby(numpy.arange(len(value_df)) // chunk_size):
        print("Processing chunk", idx + 1)
        values = value_chunk.to_dict(orient="records")
        value_sql = "(%(datetime{idx})s, %(volc{idx})s, %(sta1{idx})s, %(sta2{idx})s, %(dvv{idx})s, %(coh{idx})s, %(err{idx})s)"
    
        sql_placeholders = []
        args = {}
        for idx, value_dict in enumerate(values):
            sql_placeholders.append(value_sql.format(idx=idx))
            for key, value in value_dict.items():
                key += str(idx)
                args[key] = value
    
        SQL = "INSERT INTO wct (datetime,volc,sta1,sta2,dvv,coh,error) VALUES\n"
        SQL += ",\n".join(sql_placeholders)
        # SQL += value_sql
        SQL += """
        ON CONFLICT (datetime,volc,sta1,sta2) DO UPDATE SET
        dvv=EXCLUDED.dvv,
        coh=EXCLUDED.coh,
        error=EXCLUDED.error"""
    
        with DBCursor() as cursor:
            try:
                cursor.execute(SQL, args)
            except Exception as e:
                print(e)
    
            cursor.connection.commit()


if __name__ == "__main__":
    run_compute()
