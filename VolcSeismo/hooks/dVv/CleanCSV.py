import psycopg
import re

from pathlib import Path

from dateutil.parser import parse

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

def clean_output(pairdir, sta1, sta2):
    date_re = re.compile("([0-9-]+\s[0-9_]+)-([0-9-]+\s[0-9_]+)")

    with DBCursor() as cursor:
        cursor.execute(
            "SELECT datetime FROM wct WHERE %s::integer[] <@ sta_pair ORDER BY datetime DESC limit 1",
            ([sta1, sta2],),
        )
        try:
            start_time = cursor.fetchone()[0]
        except TypeError:
            #  No data for this station pair
            return  #  Don't do anything if we couldn't get a start date.
        start_time = start_time.replace(
            tzinfo=None
        )  # Use naieve dates. We're working in UTC here.

    for csv_file in pairdir.glob("*.csv"):
        filename = csv_file.name
        file_dates = date_re.search(filename)
        file_end = file_dates.group(2)
        file_end = file_end.replace('_', ":")
        if file_end.endswith(":"):
            file_end = file_end[:-1]
        file_end = parse(file_end)
        if file_end < start_time:
            print("Removing ", filename)
            csv_file.unlink()


if __name__ == "__main__":
    sta1 = 26
    sta2 = 28
    dirname = '/Users/israel/Development/volcano_seismology/VolcSeismo/hooks/dVv/processing/Okmok/Output/WCT/01/001_days/zz/'
    parent = Path(dirname)
    for pair_dir in parent.iterdir():
        if not pair_dir.is_dir():
            continue
        clean_output(pair_dir, sta1, sta2)