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
    
    # Step 3: Define the window size for analysis and overlap
    window_size_seconds = 60
    overlap_percent = 0
    
    sampling_rate = stream[0].stats.sampling_rate
    window_size = int(window_size_seconds * sampling_rate)
    overlap = int(window_size * overlap_percent)
    
    # Convert the stream to a DataFrame with the channel component as the header
    datas = pandas.DataFrame({
        st.meta['channel'][-1]: st.data for st in stream
    })
    
    # The start indexes for our windows
    idxs = np.arange(0, len(datas) - window_size, step = window_size - overlap)
    
    # Apply the shannon_entropy algorithim to each window, across all channels
    # Result is a DataFrame containing entropies for all times
    entropies = pandas.DataFrame([
        datas[i:i+window_size].apply(shannon_entropy)
        for i in idxs
    ])

    # Use the times from the middle of each window as our entropy times
    entropies['times'] = times[idxs + window_size // 2].values
    
    # Save the data to the database
    cursor, staid = init_db_connection(station)
    
    entropies['staid'] = staid
    
    db_data = entropies.to_dict('records')

    INSERT_SQL = """
    INSERT INTO shannon_entropy
    (station,
     entropy,
     entropy_e,
     entropy_n,
     time
    )
    VALUES (
        %(staid)s,
        %(Z)s,
        %(E)s,
        %(N)s,
        %(times)s
    )
    ON CONFLICT (station, time) DO UPDATE
    SET entropy=EXCLUDED.entropy,
        entropy_e=EXCLUDED.entropy_e,
        entropy_n=EXCLUDED.entropy_n
    """
    
    # Make sure the database is in UTC mode (it should be by default, but don't assume)
    cursor.execute("SET TIME ZONE 'UTC'")
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
    print(f"Creating entropy plot for station {station}")
    
    dest_dir = os.path.join(os.path.dirname(__file__), '../web/static/img/entropy', f"{station}.png")
    dest_dir = os.path.abspath(dest_dir)
    
    cursor.execute(
        """
        SELECT entropy,time
        FROM shannon_entropy
        WHERE station=%s
        AND entropy>0
        AND entropy!='NaN'
        ORDER BY time""",
        (staid, )
    )
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
    
    fig.savefig(dest_dir)
    plt.close(fig)
    
  
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