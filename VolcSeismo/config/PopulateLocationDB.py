from station_config import locations
# from gen_station_config import VOLCS

import psycopg2

bounds = {
    'Wrangell': {'south': 61.79160880812383,
                 'west': -144.53433413085938,
                 'north': 62.218336528147425,
                 'east': -143.50436586914063},
    'PrinceWmSn': {'south': 60.81913774332708,
                   'west': -148.24498413085936,
                   'north': 61.25933444940743,
                   'east': -147.21501586914061},
    'Susitna': {'south': 61.9874188678032,
                'west': -150.6108365234375,
                'north': 63.64814083109841,
                'east': -146.4909634765625},
    'Kantishna': {'south': 63.19574373871222,
                  'west': -151.71498413085936,
                  'north': 63.60281243515414,
                  'east': -150.6850158691406},
    'Spurr': {'south': 61.07984149686159,
              'west': -152.76888413085936,
              'north': 61.51643941253528,
              'east': -151.7389158691406},
    'Redoubt': {'south': 60.26048785010482,
                'west': -153.25878413085937,
                'north': 60.70836608469203,
                'east': -152.22881586914062},
    'Iliamna': {'south': 59.80405807663123,
                'west': -153.60678413085938,
                'north': 60.258181370619276,
                'east': -152.57681586914063},
    'Augustine': {'south': 59.304639351052046,
                  'west': -153.56374603271485,
                  'north': 59.42046182190697,
                  'east': -153.30625396728516},
    'Fourpeaked': {'south': 58.53382393072783,
                   'west': -154.18878413085937,
                   'north': 59.00517726350685,
                   'east': -153.15881586914062},
    'Katmai Region': {'south': 58.039192672527,
                      'west': -155.4682841308594,
                      'north': 58.5171944598291,
                      'east': -154.43831586914064},
    'Peulik': {'south': 57.50672638306132,
               'west': -156.88498413085938,
               'north': 57.99184614155375,
               'east': -155.85501586914063},
    'Aniakchak': {'south': 56.65677693668998,
                  'west': -158.72398413085938,
                  'north': 57.153173420646816,
                  'east': -157.69401586914063},
    'Veniaminof': {'south': 55.94418222212783,
                   'west': -159.90808413085938,
                   'north': 56.44995065263889,
                   'east': -158.87811586914063},
    'Pavlof': {'south': 55.15845074356919,
               'west': -162.40868413085937,
               'north': 55.674464033697355,
               'east': -161.37871586914062},
    'Dutton': {'south': 54.92634407479813,
               'west': -162.7893841308594,
               'north': 55.445365595034616,
               'east': -161.75941586914064},
    'Shishaldin': {'south': 54.49223754510981,
                   'west': -164.48608413085938,
                   'north': 55.01686286633674,
                   'east': -163.45611586914063},
    'Westdahl': {'south': 54.252393341421104,
                 'west': -165.1625841308594,
                 'north': 54.78010211971295,
                 'east': -164.13261586914064},
    'Akutan': {'south': 54.06644438017661,
               'west': -166.11429603271483,
               'north': 54.19960860251801,
               'east': -165.85680396728515},
    'Makushin': {'south': 53.61830303654325,
                 'west': -167.44700413085937,
                 'north': 54.154119905697264,
                 'east': -166.41703586914062},
    'Okmok': {'south': 53.14723771855238,
              'west': -168.64698413085938,
              'north': 53.68903646033407,
              'east': -167.61701586914063},
    'Cleveland': {'south': 52.54664520310331,
                  'west': -170.45998413085937,
                  'north': 53.09601847079653,
                  'east': -169.43001586914062},
    'Korovin': {'south': 52.10099042262291,
                'west': -174.66978413085937,
                'north': 52.65594594276931,
                'east': -173.63981586914062},
    'Great Sitkin': {'south': 51.79624892827805,
                     'west': -176.62588413085936,
                     'north': 52.355002679142395,
                     'east': -175.5959158691406},
    'Kanaga': {'south': 51.642995639041324,
               'west': -177.67728413085936,
               'north': 52.203653649371184,
               'east': -176.6473158691406},
    'Tanaga': {'south': 51.602544350027884,
               'west': -178.65798413085938,
               'north': 52.16370433452744,
               'east': -177.62801586914063},
    'Gareloi': {'south': 51.753709719130484,
                'west': -178.85805301635742,
                'north': 51.82400289426934,
                'east': -178.72930698364257},
    'Semisopochnoi': {'south': 51.64762440241726,
                      'west': 179.08271586914063,
                      'north': 52.20822495531319,
                      'east': -179.8873158691406},
    'Little Sitkin': {'south': 51.67207637843037,
                      'west': 178.0206158691406,
                      'north': 52.23237334622844,
                      'east': 179.05058413085936}
}

SQL = """INSERT INTO volcanoes
(site,
latitude,
longitude,
sort,
zoom,
radius,
location)
VALUES
( %(site)s,
%(latitude)s,
%(longitude)s,
%(sort)s,
%(zoom)s,
%(radius)s,
ST_SetSRID(ST_MakePoint(%(longitude)s,%(latitude)s),4326)::geography
)
ON CONFLICT (site) DO UPDATE
SET
latitude=%(latitude)s,
longitude=%(longitude)s,
sort=%(sort)s,
zoom=%(zoom)s,
radius=%(radius)s,
location=ST_SetSRID(ST_MakePoint(%(longitude)s,%(latitude)s),4326)::geography
"""

dbCon = psycopg2.connect(host = '172.16.16.206', user = 'israel',
                         password = 'Sh@nima1981', database = 'volcano_seismology')

cursor = dbCon.cursor()

for locdict in locations:
    loc = locdict['site']
    args = bounds[loc]
    args['site'] = loc
    args['radius'] = 150
    args.update(locdict)
    cursor.execute(SQL, args)

dbCon.commit()
dbCon.close()
