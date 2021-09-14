import os
import subprocess

from multiprocessing.pool import ThreadPool

import psycopg2
import yaml

BASE_SCRIPT_PATH = os.path.dirname(__file__)
R_SCRIPT_PATH = os.path.join(BASE_SCRIPT_PATH, 'calc_anomalies.R')
DEST_DIR = os.path.join(BASE_SCRIPT_PATH, 'static/img/anomalies')


def call_proc(arg):
    output = None
    try:
        output = subprocess.check_output(['Rscript', R_SCRIPT_PATH, arg, 'BHZ', DEST_DIR])
    except Exception as e:
        print(f"***ERROR generating graph for {arg}: str({e})")
    return output


if __name__ == "__main__":
    with open(os.path.join(BASE_SCRIPT_PATH, 'config.yml'), 'r') as config_file:
        config = yaml.safe_load(config_file)

    db_config = config['default']['akutan']

    os.makedirs(DEST_DIR, exist_ok = True)
    db_conn = psycopg2.connect(host=db_config['server'],
                               user = db_config['user'],
                               password = db_config['password'],
                               database = db_config['database'])
    cursor = db_conn.cursor()
    cursor.execute("""SELECT
            name
        FROM stations
        WHERE EXISTS (SELECT 1
                      FROM data
                      WHERE station=stations.id
                      LIMIT 1);
                      """)

    results = []
    pool = ThreadPool(10)
    for result in cursor:
        station = result[0]
        ret = pool.apply_async(call_proc, args = (station,))
        results.append((station, ret))

    print("Waiting for complete")
    pool.close()
    pool.join()

    db_conn.close()

    for sta, res in results:
        print("Result for", sta, ":", res.get())



