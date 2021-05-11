"""
import.py - Import a volcano seismology data file into the postgresql DB

Designed as a stand-alone script that can be run via cron or a
directory action trigger or whatever.
"""

import csv
import os

from datetime import datetime, timezone

import psycopg2


class DBCursor():
    def __init__(self, cursor_factory=None):
        self._cursor_factory = cursor_factory

    def __enter__(self):
        self._conn = psycopg2.connect(host='akutan.snap.uaf.edu', database='volcano_seismology',
                                      cursor_factory=self._cursor_factory, user="geodesy",
                                      password = 'G30dE$yU@F')
        self._cursor = self._conn.cursor()
        return self._cursor

    def __exit__(self, *args, **kwargs):
        try:
            self._conn.rollback()
        except AttributeError:
            return  # No connection
        self._conn.close()


STATION_SQL = """
SELECT
    id,
    name
FROM stations
"""
INSERT_SQL_TEMP = """
INSERT INTO data (station, channel, {})
VALUES {}
ON CONFLICT (datetime, station, channel)
DO NOTHING
"""
VALUE_TEMP = "(%s, %s, {})"
with DBCursor() as cursor:
    cursor.execute(STATION_SQL)
    STATIONS = {x[1]: x[0] for x in cursor}


def load_file(file_path, station, channel):
    with open(file_path, 'r') as file, DBCursor() as cursor:
        csv_file = csv.reader(file)
        fields = next(csv_file)
        fields[0] = "datetime"

        insert_fieldnames = ", ".join(fields)

        values = []
        insert_placeholders = []
        INSERT_STR = VALUE_TEMP.format(", ".join(["%s"] * len(fields)))
        for line in csv_file:
            point_time = datetime.strptime(line[0],
                                           '%Y-%m-%d %H:%M:%S')
            point_time = point_time.replace(tzinfo = timezone.utc)
            line[0] = point_time

            values += [station, channel] + line
            insert_placeholders.append(INSERT_STR)

        insert_placeholders = ", ".join(insert_placeholders)
        INSERT_SQL = INSERT_SQL_TEMP.format(insert_fieldnames, insert_placeholders)
        cursor.execute(INSERT_SQL, values)
        cursor.connection.commit()


if __name__ == "__main__":
    file_dir = '/Users/israel/Downloads/Volcano Seismology Data/drive-download-20210507T200936Z-001'
    files = os.listdir(file_dir)
    station = STATIONS['CLES']
    channel = 'BHZ'
    for file in files:
        file = os.path.join(file_dir, file)
        print("Loading file", file)
        load_file(file, station, channel)
