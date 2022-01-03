from VolcSeismo.process import run
from VolcSeismo.config import config

from obspy import UTCDateTime

if __name__ == "__main__":
    START = UTCDateTime("2021-12-24T16:00:09+00")
    STOP = UTCDateTime("2022-01-03T19:13:30+00")
    interval = config['GLOBAL'].getint('minutesperimage', 10)
    while START <= STOP:
        START += interval * 60
        print("Running with start value:", START)
        run(START)
    print("Data run complete")
