import logging
import os

from datetime import datetime, timedelta
from io import StringIO

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas
import psycopg2

from rpy2 import robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

try:
    from . import _process_r_vars as VARS
except ImportError:
    import _process_r_vars as VARS


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
                        events = gen_event_data(features, script_path)
                    except:
                        continue
                    # Remove any NaN rows
                    events.dropna(subset = ['begin_event', 'end_event', 'duration_event'],
                                  inplace = True)

                    # Drop any rows with the "fill" value
                    events.drop(events[events['begin_event']==-2147483648].index, inplace=True)
                    
                    if not events.empty:
                        save_events(events, station, metadata[chan])
                    create_plots(station,metadata[chan])

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

def create_plots(station,channel, date_from = None, date_to = None):
    dest_dir = os.path.join(os.path.dirname(__file__), '../web/static/img/events', f"{station}-{channel}.png")
    dest_dir = os.path.abspath(dest_dir)
    
    if date_to is None:
        date_to = datetime.utcnow()
    
    if date_from is None:
        date_from = date_to - timedelta(days = 31)
        
    cursor,sta_id=init_db_connection(station)
    
    SQL = """
    SELECT
	date,
	coalesce(events,0) as count
	FROM (
		SELECT 
			date as edate, 
			avg(events) as events
		FROM (
			SELECT 
				count(*) as events, 
				date_trunc('day',event_begin) as date, 
				ensemble 
			FROM events 
			WHERE event_begin>%(datefrom)s 
			AND event_begin<%(dateto)s 
			AND station=%(station)s
			AND channel=%(channel)s 
			GROUP BY 2 ,ensemble) s1 
		GROUP BY date) s2
	RIGHT OUTER JOIN (
		SELECT generate_series(%(datefrom)s::date, %(dateto)s::date, '1 day'::interval) as date
	) s3
	ON s3.date=s2.edate
    ORDER BY date;
    """
    params = {'datefrom': date_from,
              'dateto': date_to,
              'station': sta_id,
              'channel': channel,}
    
    # cursor.execute(SQL, params)

    events_per_day = pandas.read_sql_query(SQL, cursor.connection,
                                           params = params)
    
    fig, ax = plt.subplots()
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of events per day')
    ax.plot_date(events_per_day['date'], events_per_day['count'], 'black')
    ax.set_xlim([date_from, date_to])
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    
    fig.set_size_inches(7.5, 4)
    fig.set_dpi(300)
    
    fig.savefig(dest_dir)
    
    
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

    events['ensemble'] = events['ensemble'].astype(int)
    events['duration'] = events['duration'].astype(int)
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
    cursor.copy_from(buffer, 'events', sep = ',', columns = events.columns)

    cursor.connection.commit()

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
    create_plots('ILS', 'BHZ')

