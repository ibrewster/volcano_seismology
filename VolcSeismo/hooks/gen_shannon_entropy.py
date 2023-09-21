"""gen_shannon_entropy.py Generate Shannon entropy data."""

import logging
import os

import numpy as np
import pandas
import psycopg2
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

try:
    from . import _process_r_vars as VARS
except ImportError:
    import _process_r_vars as VARS
    
    
def run(stream, times, station, metadata):
    stream.filter('bandpass', freqmin=1.0, freqmax=16)
    
    # if not (~np.isnan(data)).all():
        # return #bad data for this site. We need not NaN
    
    
    # Step 3: Define the window size for analysis and overlap
    window_size_minutes = 1
    overlap_percent = 0
    
    # Convert minutes to seconds
    window_size_seconds = window_size_minutes * 60
    
    sampling_rate = stream[0].stats.sampling_rate
    window_size = int(window_size_seconds * sampling_rate)
    overlap = int(window_size * overlap_percent)
    
    channels = ('Z', 'E', 'N')
    datas = {
        C: stream.select(component = C).pop().data
        for C in channels
    }
    
    entropies = {C: [] for C in channels}
    
    idxs = np.arange(0, len(datas['Z']) - window_size, step = window_size - overlap)
    entropy_times = times[idxs + window_size // 2]
    
    for chan in ('Z', 'E', 'N'):
        for i in idxs:
            window_data = datas[chan][i:i+window_size]
            entropy = shannon_entropy(window_data)
            entropies[chan].append(entropy)
        
    # Save the data to the database
    cursor, staid = init_db_connection(station)
    
    db_data = [(staid, ) + x for x in zip(entropies['Z'], entropies['E'], entropies['N'], entropy_times)]

    INSERT_SQL = """
    INSERT INTO shannon_entropy
    (station,
     entropy,
     entropy_e,
     entropy_n,
     time
    )
    VALUES (%s, %s, %s, %s, %s)
    ON CONFLICT (station, time) DO UPDATE
    SET entropy=EXCLUDED.entropy
    """
    
    try:
        cursor.executemany(INSERT_SQL, db_data)
    except psycopg2.errors.CheckViolation:
        CREATE_TABLE = f"""
        CREATE TABLE entropy_parts.{station.lower()}_entropy
        PARTITION OF shannon_entropy
        FOR VALUES IN ({staid})
        """
        cursor.connection.rollback()
        cursor.execute(CREATE_TABLE)
        cursor.executemany(INSERT_SQL, db_data)
    finally:        
        cursor.connection.commit()
    
    # Create a plot of the data
    create_plot(station, staid, cursor)
    cursor.connection.close()

def create_plot(station, staid, cursor):
    logging.info(f"Creating entropy plot for station {station}")
    print(f"**Creating entropy plot for station {station}")
    
    dest_dir = os.path.join(os.path.dirname(__file__), '../web/static/img/entropy', f"{station}.png")
    dest_dir = os.path.abspath(dest_dir)
    
    cursor.execute("SELECT entropy,time FROM shannon_entropy WHERE station=%s AND entropy>0 ORDER BY time", (staid, ))
    plot_data = cursor.fetchall()
    entropies, times = list(zip(*plot_data))
    
    fig, ax = plt.subplots()
    
    fig.autofmt_xdate()
    ax.set_xlabel('Date')
    ax.set_ylabel('Shannon Entropy')    

    ax.plot(times, entropies, linestyle = 'solid', color = 'blue')
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%y %H:%M'))
    
    fig.set_size_inches(12, 4)
    fig.set_dpi(300)
    fig.tight_layout()
    
    logging.info(f"Saving entropy plot for station {station}")
    print(f"**Saving entropy plot for station {station}")
    
    fig.savefig(dest_dir)
    plt.close(fig)
    print(f"!!!!Entropy plot saved for station {station}")
    
  
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

# Function to calculate Shannon entropy
def shannon_entropy(data):
    params = stats.norm.fit(data)
    dist = stats.norm(*params)
    probabilities = dist.pdf(data)
    sh_entropy = -np.sum(probabilities * np.log2(probabilities))
    return sh_entropy