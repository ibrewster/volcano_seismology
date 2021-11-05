from cachetools.func import ttl_cache
import psycopg2
from psycopg2.extras import RealDictCursor


class DBCursor():
    def __init__(self, cursor_factory=None):
        self._cursor_factory = cursor_factory

    def __enter__(self):
        self._conn = psycopg2.connect(host='137.229.113.120',
                                      database='volcano_seismology',
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


@ttl_cache()  # Default cache length 10 minutes
def _get_data():
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
        volc = dict(volc)  # Make the RealDict a Real Dict.
        stations = volc['stations']
        site = volc['site']
        del volc['site']
        for station in stations:
            station[1] = float(station[1])

        volcs[site] = volc

    return volcs


class Locations:
    def __len__(self):
        return len(_get_data())

    def __getitem__(self, item):
        return _get_data()[item]

    def __iter__(self):
        return iter(_get_data())

    def __str__(self):
        return str(_get_data())

    def __repr__(self):
        return repr(_get_data())

    def __getattr__(self, name):
        return getattr(_get_data(), name)
