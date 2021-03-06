import configparser
import operator
import os

from functools import reduce

config_dir = os.path.dirname(__file__)


def main():
    config = configparser.ConfigParser(allow_no_value=True)
    config['GLOBAL'] = {
        'MinutesPerImage': 10,
    }

    config['WINSTON'] = {
        '; Currently waveform data is pulled from a winston.': None,
        '; In the future, this may become configurable to any ObsPy data source': None,
        'url': 'pubavo1.wr.usgs.gov',
        'port': 16022,
    }

    config['MySQL'] = {
        '; MySQL is used to pull latitude/longitude information for volcanos': None,
        "; from geodiva so Cheryl Cameron doesn't have my head": None,
        'DB_HOST': 'spurr.snap.uaf.edu',
        'DB_USER': '<MY_USER>',
        'DB_PASSWORD': '<MY_PASSWORD>',
        'DB_NAME': 'geodiva',
    }

    config['IRIS'] = {'url': 'https://service.iris.edu/fdsnws/station/1/query', }

    # Data filters to apply to the raw data
    config['FILTER'] = {
        'LowCut': .5,
        'HighCut': 15,
        'Order': 2,
    }

    # Parameters for generating the seismic spectrogram function
    config['SPECTROGRAM'] = {
        'WindowType': 'hamming',
        'WindowSize': 1024,
        'Overlap': 924,
        'NFFT': 1024,
        'MaxFreq': 10,
        'MinFreq': 0,
    }

    # Get any enviroment variable overrides
    for key in config:
        for envkey, value in os.environ.items():
            if envkey.startswith(key):
                print("Override found for key", key, ":", envkey)
                env_parts = envkey.split('__')
                try:
                    reduce(operator.getitem, env_parts[:-1], config)[env_parts[-1]] = value
                except:
                    print("Unable to get dest for override key", envkey)

    with open(os.path.join(config_dir, 'config.ini'), 'w') as conffile:
        config.write(conffile)


if __name__ == "__main__":
    main()
