import configparser
import os

import psycopg2

from cachetools.func import ttl_cache
from psycopg2.extras import RealDictCursor


class DBCursor():
    def __init__(self, cursor_factory=None):
        self._cursor_factory = cursor_factory
        config = configparser.ConfigParser()
        config.read(os.path.join(os.path.dirname(__file__), "config.ini"))
        pg_config = config['PostgreSQL']

        self._db_host = pg_config['db_host']
        self._db_user = pg_config['db_user']
        self._db_password = pg_config['db_password']
        self._db_name = pg_config['db_name']

    def __enter__(self):
        self._conn = psycopg2.connect(host=self._db_host,
                                      database=self._db_name,
                                      cursor_factory=self._cursor_factory,
                                      user=self._db_user,
                                      password = self._db_password)
        self._cursor = self._conn.cursor()
        return self._cursor

    def __exit__(self, *args, **kwargs):
        try:
            self._conn.rollback()
        except AttributeError:
            return  # No connection
        self._conn.close()


@ttl_cache()  # Default cache length 10 minutes
def _get_location_data():
    SQL = """
    SELECT site,latitude,longitude,sort,zoom, 
array((SELECT array[name,dist::text]
       FROM 
         (SELECT name, (stations.location <-> volcanoes.location)/1000 as dist
          FROM stations) s1
     WHERE dist<=volcanoes.radius
     ORDER BY dist
     LIMIT 10)) as stations 
FROM volcanoes;
    """
    with DBCursor(RealDictCursor) as cursor:
        cursor.execute(SQL)
        locations = cursor.fetchall()

    # Process locations into desired table
    volcs = {}
    for volc in locations:
        volc = dict(volc)  # Make the RealDict a REAL Dict.
        stations = volc['stations']
        site = volc['site']
        del volc['site']
        for station in stations:
            station[1] = float(station[1])

        volcs[site] = volc

    return volcs


@ttl_cache()  # Default cache length 10 minutes
def _get_station_data():
    SQL = """
    SELECT
        name,
        chan,
        net,
        sample_rate,
        scale,
        latitude,
        longitude
    FROM stations;
    """
    with DBCursor(RealDictCursor) as cursor:
        cursor.execute(SQL)
        stations = cursor.fetchall()

    # Process locations into desired table
    stns = {}
    for station in stations:
        sta_name = station['name']
        station = dict(station)  # Make the RealDict a REAL Dict.
        del station['name']
        for key in ['chan', 'net', 'sample_rate', 'scale']:
            station[key.upper()] = station[key]
            del station[key]

        stns[sta_name] = station

    return stns


class Locations:
    def __len__(self):
        return len(_get_location_data())

    def __getitem__(self, item):
        return _get_location_data()[item]

    def __iter__(self):
        return iter(_get_location_data())

    def __str__(self):
        return str(_get_location_data())

    def __repr__(self):
        return repr(_get_location_data())

    def __getattr__(self, name):
        return getattr(_get_location_data(), name)


class Stations:
    def __len__(self):
        return len(_get_station_data())

    def __getitem__(self, item):
        return _get_station_data()[item]

    def __iter__(self):
        return iter(_get_station_data())

    def __str__(self):
        return str(_get_station_data())

    def __repr__(self):
        return repr(_get_station_data())

    def __getattr__(self, name):
        return getattr(_get_station_data(), name)
