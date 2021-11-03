from datetime import timedelta
import psycopg2
from psycopg2.extras import RealDictCursor

from . import app


class db_cursor():
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


def load_stations(max_age=10):
    # Load station list from DB
    SQL = """
    SELECT
        name,
        latitude::float as lat,
        longitude::float as lng,
        site,
        id as sta_id,
        channels
    FROM stations
    INNER JOIN station_channels ON station_channels.station=stations.id
    WHERE EXISTS (SELECT 1
                  FROM last_data
                  WHERE station=stations.id
                  AND lastdata>now()-%s
                  LIMIT 1);
    """

    try:
        max_age = timedelta(days = 365.25 * max_age)
    except ValueError:
        max_age = timedelta(days = 365.25 * 100)  # just some large random time in the past

    try:
        with db_cursor(cursor_factory = RealDictCursor) as cursor:
            cursor.execute(SQL, (max_age, ))
            stas = {row['name']: dict(row)
                    for row in cursor}
    except:
        app.logger.exception("Unable to load stations from db")

    return stas


class FetchingDict(dict):
    def __getitem__(self, key):
        if not self:
            print("Loading stations from DB")
            self.update(load_stations(10))
        return super().__getitem__(key)

    def fetch(self, age = 10):
        self.clear()
        self.update(load_stations(age))


stations = FetchingDict()
