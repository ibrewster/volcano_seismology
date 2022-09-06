import os
import subprocess
import time

from datetime import datetime
from multiprocessing.pool import ThreadPool

import psycopg2
import yaml

BASE_SCRIPT_PATH = os.path.dirname(__file__)
R_SCRIPT_PATH = os.path.join(BASE_SCRIPT_PATH, 'calc_anomalies.R')
DEST_DIR = os.path.join(BASE_SCRIPT_PATH, 'static/img/anomalies')


def call_proc(station, channel):
    output = None
    try:
        output = subprocess.check_output(['Rscript', R_SCRIPT_PATH, station, channel, DEST_DIR])
    except Exception as e:
        print(f"***ERROR generating graph for {station}, {channel}: str({e})")
    return output


if __name__ == "__main__":
    start_t = time.time()
    with open(os.path.join(BASE_SCRIPT_PATH, 'config.yml'), 'r') as config_file:
        config = yaml.safe_load(config_file)

    db_config = config['default']['DATABASE']

    os.makedirs(DEST_DIR, exist_ok = True)
    db_conn = psycopg2.connect(host=db_config['server'],
                               user = db_config['user'],
                               password = db_config['password'],
                               database = db_config['database'])
    cursor = db_conn.cursor()
    
    # Alternate SELECT not using last_data. *slightly* slower (not enough to matter),
    # But doesn't rely on last_data being up-to-date
    #     """SELECT name,channels FROM
    # (SELECT
    # 	name,
    # 	channels,
    # 	(SELECT true FROM data WHERE station=stations.id LIMIT 1) as has_data
    # FROM stations
    # INNER JOIN station_channels ON station_channels.station=stations.id) s1
    # WHERE s1.has_data=true"""

    cursor.execute("""SELECT
	name, channels
FROM stations
INNER JOIN station_channels
ON station_channels.station=stations.id
WHERE EXISTS (SELECT 1
	FROM last_data
	WHERE station=stations.id
	LIMIT 1);
        """)

    results = []
    pool = ThreadPool(12)
    db_data = cursor.fetchall()
    db_conn.close()

    for row_data in db_data:
        station = row_data[0]
        channels = row_data[1]
        for channel in channels:
            if not channel.endswith('HZ'):
                continue
            ret = pool.apply_async(call_proc, args = (station, channel))
            results.append((station, channel, ret))

    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}Waiting for complete")
    pool.close()
    pool.join()

    for sta, chan, res in results:
        print("Result for", sta, chan, ":", res.get())

    print(f"Completed run in {(time.time()-start_t)/60} Minutes")

