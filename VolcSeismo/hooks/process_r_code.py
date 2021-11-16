import logging
import os

from io import StringIO

import pandas
import psycopg2

from rpy2 import robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

from . import _process_r_vars as VARS


def run(data, station, metadata):
    base = importr('base')

    # Load the R script
    script_path = os.path.join(os.path.dirname(__file__), 'VolcSeismo.R')
    base.source(script_path)
    r_func = robjects.globalenv['runAnalysis']
    with localconverter(robjects.default_converter + pandas2ri.converter):
        for chan in ('Z', 'E', 'N'):
            if metadata[chan]:
                result = r_func(data['time'], data[chan])
                save_to_db(result, station, metadata[chan])
    return result


def save_to_db(data, station, channel = 'BHZ'):
    if len(data) == 0:
        print("NOT saving result for", station, channel, "No data provided")
        return

    conn = psycopg2.connect(host = VARS.DB_HOST,
                            database = 'volcano_seismology',
                            user = VARS.DB_USER,
                            password = VARS.DB_PASSWORD)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM stations WHERE name=%s", (station, ))
    sta_id = cursor.fetchone()
    if sta_id is None:
        print("Unable to store result for", station, ". No station id found.")
        return

    print("Saving result for", station, channel)
    data.replace('', '\\N', inplace = True)
    sta_id = sta_id * len(data)
    data['station'] = sta_id
    data['channel'] = channel
    data.rename(columns = {'V1': 'datetime'}, inplace = True)
    data['datetime'] = pandas.to_datetime(data['datetime'],
                                          infer_datetime_format = True,
                                          utc = True).astype('datetime64[ns, UTC]')

    try:
        t_start = data.datetime.min()
    except Exception as e:
        logging.warning(f"The value of the datetime column is: {data.datetime}")
        logging.exception(f"Exception getting min datetime ({e}):\n")
        raise

    t_stop = data.datetime.max()

    # Set DB to UTC so we don't have time zone issues
    cursor.execute("SET timezone = 'UTC'")

    # Delete any records covered by this run
    DEL_SQL = """
    DELETE FROM data
    WHERE station=%s
    AND channel=%s
    AND datetime>=%s
    AND datetime<=%s
    """
    cursor.execute(DEL_SQL, (sta_id[0], channel, t_start, t_stop))

    buffer = StringIO()
    # Drop any duplicate values so we can do an insert
    data.drop_duplicates(('datetime', 'station', 'channel'), keep = 'last', inplace = True,
                         ignore_index = True)
    data.to_csv(buffer, index = False, header = False,
                na_rep = '\\N')
    buffer.seek(0)

    cursor.copy_from(buffer, 'data', sep = ",", columns = data.columns)
    conn.commit()
    cursor.close()


# The following is just for debugging purposes, to run this hook asside from other processing.
if __name__ == "__main__":
    import pandas

    # test file
    df = pandas.read_csv("WACK_2021_06_03_14_50_09.csv")
    result = run(df)
    result.to_csv('pythonResult.csv')

