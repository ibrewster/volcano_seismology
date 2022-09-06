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
    gen_event_data = robjects.globalenv['genEventGraphs1']
    graph_events_day = robjects.globalenv['EventsPerDay']
    with localconverter(robjects.default_converter + pandas2ri.converter):
        for chan in ('Z', 'E', 'N'):
            if metadata[chan]:
                try:
                    features = r_func(data['time'], data[chan], station, chan, script_path)
                except:
                    continue

                if chan == 'Z':
                    try:
                        events = gen_event_data(features, station, script_path)
                    except:
                        continue
                    # Remove any NaN rows
                    events.dropna(subset = ['begin_event', 'end_event', 'duration_event'],
                                  inplace = True)
                    if not events.empty:
                        #plot_results = graph_events_day(events, station, script_path)
                        save_events(events, station, metadata[chan])

                save_to_db(features, station, metadata[chan])
    return features


def init_db_connection(station):
    conn = psycopg2.connect(host = VARS.DB_HOST,
                            database = 'volcano_seismology',
                            user = VARS.DB_USER,
                            password = VARS.DB_PASSWORD)
    cursor = conn.cursor()
    # Set DB to UTC so we don't have time zone issues
    cursor.execute("SET timezone = 'UTC'")

    cursor.execute("SELECT id FROM stations WHERE name=%s", (station, ))
    sta_id = cursor.fetchone()
    if sta_id is None:
        return (None, None)

    return (cursor, sta_id[0])


def save_events(events, station, channel):
    print(f"Saving {len(events)} events for {station}, {channel}")
    cursor, sta_id = init_db_connection(station)

    events['station'] = sta_id
    events['channel'] = channel
    events.rename(columns = {'begin_event': 'event_begin',
                             'end_event': 'event_end',
                             'duration_event': 'duration',
                             'ampl_event': 'ampl',
                             'fre_event': 'frequency', },
                  inplace = True)

    events['ensemble'] = events['ensemble'].asttype(int)
    events['event_begin'] = pandas.to_datetime(events['event_begin'],
                                               infer_datetime_format = True,
                                               utc = True).astype('datetime64[ns, UTC]')
    events['event_end'] = pandas.to_datetime(events['event_end'],
                                             infer_datetime_format = True,
                                             utc = True).astype('datetime64[ns, UTC]')

    DEL_SQL = """
    DELETE FROM events
    WHERE station=%s
    AND channel=%s
    AND event_begin>=%s
    AND event_begin<=%s
    """
    t_start = events['event_begin'].min()
    t_stop = events['event_begin'].max()
    cursor.execute(DEL_SQL, (sta_id, channel, t_start, t_stop))

    buffer = StringIO()
    events.to_csv(buffer, index = False, header = False)
    buffer.seek(0)
    print(f"Running copy for station {station}")
    cursor.copy_from(buffer, 'events', sep = ',', columns = events.columns)

    cursor.connection.commit()
    print(f"Finished insert for station {station}")

    cursor.close()


def save_to_db(data, station, channel = 'BHZ'):
    if len(data) == 0:
        print("NOT saving result for", station, channel, "No data provided")
        return

    cursor, sta_id = init_db_connection(station)
    if cursor is None or sta_id is None:
        print("Unable to store result for", station, ". No station id found.")
        return

    conn = cursor.connection

    print("Saving result for", station, channel)
    data.replace('', '\\N', inplace = True)
    data['station'] = sta_id
    data['channel'] = channel
    data.rename(columns = {'V1': 'datetime',
                           'as.character(time_parameters)': 'datetime'},
                inplace = True)
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

    # Delete any records covered by this run
    DEL_SQL = """
    DELETE FROM data
    WHERE station=%s
    AND channel=%s
    AND datetime>=%s
    AND datetime<=%s
    """
    cursor.execute(DEL_SQL, (sta_id, channel, t_start, t_stop))

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

