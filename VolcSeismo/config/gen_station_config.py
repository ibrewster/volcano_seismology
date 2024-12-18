import configparser
import csv
import os
import pprint

from collections import defaultdict
from io import StringIO

import numpy as np
import pandas
import psycopg2
from psycopg2 import extras
import requests

from obspy import UTCDateTime
from obspy.clients.earthworm import Client as WClient

config_dir = os.path.dirname(__file__)

####################################################
#  User editable options for the station configuration
#
# Change values below to set the parameters used when
# generating the station config file
####################################################
# Search for all channels matching the below mask
CHANNEL_MASK = '[SBE]HZ'

# The default channel to use if more than one are found
DEFAULT_CHANNEL = 'BHZ'

# Which networks to search for stations
NETWORKS = ['AV', 'AK']

# Maximum number of stations to show per volcano plot
MAX_STATIONS = 10

# Only include stations that have received data within this time period (seconds)
MAX_AGE = 1 * 24 * 60 * 60  # 1 days

# Maximum distance from volcano to search for stations.
# Can be over-ridden on a per-volcano basis in the VOLCS list, below
DEFAULT_RADIUS = 150
##########################################################
# END USER SETTINGS
##########################################################


def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.

    Less precise than vincenty, but fine for short distances,
    and works on vector math

    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km


def get_meta(NET, config):
    args = {
        'net': NET,
        'level': 'channel',
        'format': 'text'
    }

    IRIS_URL = config['IRIS']['url']

    resp = requests.get(IRIS_URL, args)

    # Parse the delimited response
    resp_str = StringIO(resp.text)
    reader = csv.reader(resp_str, delimiter = '|')
    keys = [x.strip() for x in next(reader)]
    return {(row[1], row[3]): dict(zip(keys, row)) for row in reader}


def make_net_dict(avail, meta):
    nets = pandas.DataFrame()
    channels = defaultdict(list)
    for item in avail:
        sta = item[1]
        chan = item[3]
        last_data = item[5]
        last_data_age = (UTCDateTime() - last_data)  # in seconds
        if last_data_age > MAX_AGE:
            continue  # Data too old. Discard channel.

        sta_meta = meta.get((sta, chan))

        if not sta_meta:
            continue

        if sta not in channels:
            # Only append this to the data frame if we haven't seen it before

            item_dict = pandas.DataFrame(
                [{ 'station': sta, 'latitude':
                   float(sta_meta['Latitude']), 'longitude': float(sta_meta['Longitude']),
                }]
            )
            nets = pandas.concat([nets, item_dict], ignore_index=True)

        chan_data = (
            chan,
            int(float(sta_meta['Scale'])),
            float(sta_meta['SampleRate'])
        )
        channels[sta].append(chan_data)

    return nets, channels


class PostgresCursor:
    def __init__(self, cursor_factory = None):
        self._cursor_factory = cursor_factory
        config = configparser.ConfigParser()
        config.read(os.path.join(config_dir, "config.ini"))
        pg_config = config['PostgreSQL']

        self._db_host = pg_config['db_host']
        self._db_user = pg_config['db_user']
        self._db_password = pg_config['db_password']
        self._db_name = pg_config['db_name']
        self._db_superpass=pg_config['db_superpass']
        self._db_superuser=pg_config['db_superuser']

    def __enter__(self):
        self._conn = psycopg2.connect(host=self._db_host,
                                      database=self._db_name,
                                      cursor_factory=self._cursor_factory,
                                      user=self._db_superuser,
                                      password = self._db_superpass)
        self._cursor = self._conn.cursor()
        return self._cursor

    def __exit__(self, *args, **kwargs):
        try:
            self._conn.rollback()
        except AttributeError:
            return  # No connection
        self._conn.close()


def generate_stations():
    config = configparser.ConfigParser()
    config.read(os.path.join(config_dir, "config.ini"))

    with PostgresCursor(cursor_factory = extras.RealDictCursor) as cursor:
        cursor.execute("SELECT site,latitude,longitude,sort,zoom,radius FROM volcanoes")
        VOLCS = {x['site']: dict(x) for x in cursor}

    # Get volcano locations
    VOLCS_NEEDING_LOC = [key for key, value in VOLCS.items()
                         if 'latitude' not in value or value['latitude'] is None]

    # Don't try to do the query if there are no volcs needing lat/lon information
    if VOLCS_NEEDING_LOC:
        import pymysql  # Import here just in case we don't actually need it

        DB_HOST = config['MySQL']['db_host']
        DB_USER = config['MySQL']['db_user']
        DB_PASS = config['MySQL']['db_password']
        DB_NAME = config['MySQL']['db_name']

        dbconn = pymysql.connect(user = DB_USER, password = DB_PASS,
                                 host = DB_HOST, database = DB_NAME)
        cursor = dbconn.cursor()
        cursor.execute('SELECT volcano_name, latitude, longitude FROM volcano WHERE volcano_name in %s',
                       (VOLCS_NEEDING_LOC, ))

        for volc, lat, lon in cursor:
            VOLCS[volc]['latitude'] = lat
            VOLCS[volc]['longitude'] = lon

    winston_url = config['WINSTON']['url']
    winston_port = config['WINSTON'].getint('port', 16022)
    wclient = WClient(winston_url, winston_port)

    all_channels = {}
    all_nets = None
    for network in NETWORKS:
        # Get availability information for the network
        avail = wclient.get_availability(network=network, channel = CHANNEL_MASK)
        # get metadata for the network
        meta = get_meta(network, config)
        nets, channels = make_net_dict(avail, meta)
        nets['net'] = [network] * len(nets)
        if all_nets is None:
            all_nets = nets
        else:
            all_nets = pandas.concat([all_nets, nets])

        all_channels.update(channels)

    locations = []
    stations = {}
    for volc, info in VOLCS.items():
        lat1 = info['latitude']
        lon1 = info['longitude']
        info['sort'] = lon1 if lon1 < 0 else lon1 - 360
        info['zoom'] = info.get('zoom', 10)  # basically, replace none with default
        locations.append(info)

        # Figure out which sites are within interest distance of a volcano
        all_nets['dist'] = haversine_np(lon1, lat1, all_nets.longitude, all_nets.latitude)
        max_dist = info.get('radius', DEFAULT_RADIUS)
        avail_nets = all_nets.loc[all_nets.dist <= max_dist]
        chosen_nets = avail_nets.sort_values('dist').head(MAX_STATIONS)
        for net in chosen_nets.itertuples():
            sta = net.station
            if sta not in stations:
                channels, scales, rates = zip(*all_channels[sta])
                sta_dict = {
                    'name': sta,
                    'net': net.net,
                    'longitude': net.longitude,
                    'latitude': net.latitude,
                }

                if DEFAULT_CHANNEL not in channels:
                    sta_dict['chan'] = channels[0]
                    sta_dict['scale'] = scales[0]
                    sta_dict['sample_rate'] = rates[0]
                else:
                    chan_idx = channels.index(DEFAULT_CHANNEL)
                    sta_dict['chan'] = DEFAULT_CHANNEL
                    sta_dict['scale'] = scales[chan_idx]
                    sta_dict['sample_rate'] = rates[chan_idx]

                stations[sta] = sta_dict

    VOLCANO_SQL = """
    UPDATE volcanoes SET
        latitude = %(latitude)s,
        longitude = %(longitude)s,
        sort = %(sort)s,
        zoom = %(zoom)s,
        radius = %(radius)s
    WHERE site = %(site)s
    """

    STATION_SQL = """
    INSERT INTO stations (
        name,
        latitude,
        longitude,
        chan,
        net,
        sample_rate,
        scale
    )
    VALUES (
        %(name)s,
        %(latitude)s,
        %(longitude)s,
        %(chan)s,
        %(net)s,
        %(sample_rate)s,
        %(scale)s
    )
    ON CONFLICT (name) DO UPDATE SET
    latitude=%(latitude)s,
    longitude=%(longitude)s,
    chan=%(chan)s,
    net=%(net)s,
    sample_rate=%(sample_rate)s,
    scale=%(scale)s
    RETURNING id
    """

    stations = list(stations.values())
    with PostgresCursor() as cursor:
        print("Updating volcano metadata")
        cursor.executemany(VOLCANO_SQL, locations)

        for station_data in stations:
            print("Updating entry for", station_data['name'])
            cursor.execute(STATION_SQL, station_data)
            cursor.connection.commit()
            # Make sure we have data tables for all stations/channels.
            staid = cursor.fetchone()[0]
            name = station_data['name']
            table_name = f"data_{name}"

            TABLE_SQL = f"""CREATE TABLE IF NOT EXISTS data_parts.{table_name}
            PARTITION OF data
            FOR VALUES IN ({staid})
            PARTITION BY LIST (channel)
            TABLESPACE pool
            """
            cursor.execute(TABLE_SQL)

            primary_channel = station_data['chan']

            # Make sure data tables exist for all channels.
            for suffix in ['E', 'N', 'Z']:
                channel = primary_channel[:-1] + suffix
                channel_table_name = f"{table_name}_{channel}"
                print("Creating channel table for", channel_table_name)

                CHANNEL_SQL = f"""
                CREATE TABLE IF NOT EXISTS data_parts.{channel_table_name}
                PARTITION OF data_parts.{table_name}
                FOR VALUES IN ('{channel}')
                WITH (autovacuum_vacuum_insert_scale_factor=0.005, autovacuum_freeze_min_age=0, autovacuum_analyze_scale_factor='0.005')
                TABLESPACE pool
                """
                cursor.execute(CHANNEL_SQL)

        print("SQL Complete.")
        cursor.connection.commit()

    with open(os.path.join(config_dir, 'station_config.py'), 'w') as conf_file:
        conf_file.write('"""\nThis file is automatically generated by running gen_station_config.py\n')
        conf_file.write('You may modify this file if desired, but be aware any changes WILL\n')
        conf_file.write('be over-written the next time gen_station_config is run\n"""')
        conf_file.write('\n\n')
        pprinter = pprint.PrettyPrinter(stream = conf_file)
        conf_file.write('locations=')
        pprinter.pprint(locations)
        conf_file.write('\n\n')
        conf_file.write('stations=')
        pprinter.pprint(stations)


if __name__ == "__main__":
    generate_stations()
