import logging
import os
import sqlite3
import tempfile

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

try:
    from .config import config, stations
    from .hooks import run_hooks
    from .waveform import load
except ImportError:
    # Not running as a module
    from config import config, stations
    from hooks import run_hooks
    from waveform import load

from obspy import UTCDateTime


def run():
    print("running")

    # Set endtime to the closest 10 minute mark prior to current time
    ENDTIME = UTCDateTime()
    ENDTIME = ENDTIME.replace(minute=ENDTIME.minute - (ENDTIME.minute % 10),
                              second=0,
                              microsecond=0)

    STARTTIME = ENDTIME - (config['GLOBAL'].getint('minutesperimage', 10) * 60)

    gen_times = [(STARTTIME, ENDTIME, None)]

    # Gather any "missed" time segments to retry
    cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
    os.makedirs(cache_dir, exist_ok = True)
    cache_db = os.path.join(cache_dir, 'cache.db')
    with sqlite3.connect(cache_db) as conn:
        cur = conn.cursor()
        # Make sure the "missed" table exists
        cur.execute("CREATE TABLE IF NOT EXISTS missed (station TEXT, dtfrom TEXT, dtto TEXT, UNIQUE (station, dtfrom, dtto))")
        cur.execute("SELECT station,dtfrom,dtto FROM missed ORDER BY dtto DESC")  # May be empty

        for station, dtfrom, dtto in cur:
            try:
                loc = {station: stations[station]}
            except KeyError:
                pass  # No metadata for this station, can't be processed.

            dtfrom = UTCDateTime(dtfrom)
            dtto = UTCDateTime(dtto)
            if (UTCDateTime() - dtfrom) / 60 / 60 > 2:
                continue  # Don't try to go back more than two hours
            gen_times.append((dtfrom, dtto, loc))

        cur.execute("DELETE FROM missed")  # Potential race condition between SELECT and DELETE?

    # Run the hook scripts
    procs = []
    with ProcessPoolExecutor() as executor:
        for start, end, locs in gen_times:
            # Create a temporary directory that we can use to avoid duplicating effort
            # if the same station appears for multiple volcanos.
            with tempfile.TemporaryDirectory() as tempdir:
                print("Created temporary directory", tempdir)
                if locs is None:
                    locs = stations  # All stations

                    #############DEBUG####################
                    locs = {'ANNE': stations['ANNE'], }
                    ######################################
                for loc, loc_info in locs.items():
                    lock_file = os.path.join(tempdir, loc)
                    if os.path.exists(lock_file):
                        print(f"!!!ALREADY RAN STATION {station}. SKIPPING!!!!")
                        continue

                    # Touch a file to indicate this station has been processed for this time range
                    Path(lock_file).touch()

                    # And do it
                    future = executor.submit(_process_data, loc,
                                             loc_info, start, end)
                    procs.append((loc, start, end, future))
            ############DEBUG###############
            break
            #################################

    # process the results
    INSERT_SQL = """
    INSERT INTO missed
    (station, dtfrom, dtto)
    VALUES
    (?,?,?)
    """
    for station, dtstart, dtend, proc in procs:
        try:
            missed_flag = proc.result()
        except Exception as e:
            print(e)
            missed_flag = True

        if missed_flag:
            with sqlite3.connect(cache_db) as conn:
                cur = conn.cursor()
                try:
                    cur.execute(INSERT_SQL,
                                (station,
                                 dtstart.isoformat(),
                                 dtend.isoformat()))
                except sqlite3.IntegrityError:
                    # Already marked this volc/timerange as missing
                    continue

                conn.commit()


def _process_data(STA, sta_dict, STARTTIME, ENDTIME):
    CHAN = sta_dict.get('CHAN', 'BHZ')
    NET = sta_dict.get('NET', 'AV')

    stream, waveform_times = load(NET, STA, '--', CHAN, STARTTIME, ENDTIME)
    if stream is None or waveform_times is None:
        logging.warning(f"No data retrieved for station {STA}, {STARTTIME} to {ENDTIME}")
        return True  # missed this station/time

    run_hooks(stream, waveform_times)


if __name__ == "__main__":
    ENDTIME = UTCDateTime('2020-09-14T16:00:00')
    STARTTIME = UTCDateTime('2020-09-14T02:40:00')
    gen_times = []
    start = STARTTIME
    end = STARTTIME
    while end < ENDTIME:
        end = start + (config['GLOBAL'].getint('minutesperimage', 10) * 60)
        gen_times.append((start, end, None))
        start = end

    print(gen_times)
    procs = []
    with ProcessPoolExecutor() as executor:
        for start, end, locs in gen_times:
            with tempfile.TemporaryDirectory() as tempdir:
                print("Created temporary directory", tempdir)
                if locs is None:
                    locs = stations  # All stations
                for loc, loc_info in locs.items():
                    lock_file = os.path.join(tempdir, loc)
                    if os.path.exists(lock_file):
                        print(f"!!!ALREADY RAN STATION {station}. SKIPPING!!!!")
                        continue

                    # Touch a file to indicate this station has been processed for this time range
                    Path(lock_file).touch()

                    # And do it
                    future = executor.submit(_process_data, loc,
                                             loc_info, start, end)
                    procs.append((loc, start, end, future))

    for station, dtstart, dtend, proc in procs:
        try:
            missed_flag = proc.result()
        except Exception as e:
            print(e)
            missed_flag = True
