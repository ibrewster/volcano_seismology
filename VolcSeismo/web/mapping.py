import csv
import glob
import os
import time
import json
import uuid

from datetime import datetime, timedelta, timezone
from io import StringIO

import pandas

from . import app, compressor
from . import utils
from .utils import stations

from VolcSeismo.config import locations

import ujson
import json
import numpy
import flask
import flask.helpers

from dateutil.parser import parse
from PIL import Image

import pandas as pd
import plotly.graph_objects as go


def volc_sort_key(volc):
    name, data = volc
    return data['sort']


@app.route("/")
def _map():
    flask.session['public'] = False

    sort_order = [key for key, value in
                  sorted(locations.items(),
                         key = volc_sort_key,
                         reverse = True)]
    args = {'locations': locations, 'volcOrder': sort_order, }
    return flask.render_template("index.html", **args)


def genStationColors(row):
    if row.type == "continuous":
        return "73/0/244"

    if row.type == "campaign":
        return "12/218/59"


@app.route('/gen_graph', methods = ["GET"])
@compressor.compressed()
def get_graph_image():
    try:
        date_from = parse(flask.request.args.get('dfrom'))
    except:
        date_from = None

    date_to = parse(flask.request.args['dto'])
    station = flask.request.args['station']
    channel = flask.request.args['channel']
    fmt = flask.request.args.get('fmt', 'png')
    width = int(flask.request.args.get('width', 900))

    title = station

    title += ' - '
    data = get_graph_data(False, station=station, date_from=date_from, date_to=date_to)

    if not data['dates']:
        return "no data found", 404

    if date_from is None:
        date_from = parse(data['dates'][0] + "T00:00:00Z")

    freq_max = gen_plot_data_dict(data['dates'], data['freq_max10'])
    sd_freq_max = gen_plot_data_dict(data['dates'], data['sd_freq_max10'], 2)
    rsam = gen_plot_data_dict(data['dates'], data['rsam'], 3)

    plot_data = [freq_max, sd_freq_max, rsam]

    layout = gen_subgraph_layout(plot_data,
                                 ['Freq Max10 (Hz)', 'SD Freq Max10 (Hz)',
                                  'RSAM (counts)'],
                                 date_from, date_to)
    layout['annotations'] = [{
        "xref": 'paper',
        "yref": 'paper',
        "x": 0.004,
        "xanchor": 'left',
        "y": 1.005,
        "yanchor": 'bottom',
        "text": f"Channel: {channel}",
        "showarrow": False,
        "font": {"size": 12}
    }]

    # title += f'{date_from.strftime("%Y-%m-%d")} to {date_to.strftime("%Y-%m-%d")}'
    # layout['title'] = {'text': title,
    # 'x': .115,
    # 'y': .939,
    # 'xanchor': 'left',
    # 'yanchor': 'bottom',
    # 'font': {
    # 'size': 16,
    # }, }
    return gen_graph_image(plot_data, layout, fmt, 'inline',
                           width = width)


def gen_plot_data_dict(x, y, idx=None):
    trace = {
        "x": x,
        "y": y,
        "type": 'scatter',
        "mode": 'markers',
        "marker": {
            "size": 2,
            "color": 'rgb(55,128,256)'
        },
    }

    if idx is not None:
        trace['yaxis'] = f'y{idx}'
        trace['xaxis'] = f'x{idx}'

    return trace


def gen_subgraph_layout(data, titles, date_from, date_to):
    if not isinstance(titles, (list, tuple)):
        titles = [titles, ]

    script_path = os.path.realpath(os.path.dirname(__file__))
    LOGO_PATH = os.path.join(script_path, 'static/img/logos.png')
    logo = Image.open(LOGO_PATH)

    layout = {
        "paper_bgcolor": 'rgba(255,255,255,1)',
        "plot_bgcolor": 'rgba(255,255,255,1)',
        "showlegend": False,
        "margin": {
            "l": 50,
            "r": 25,
            "b": 25,
            "t": 80,
            "pad": 0
        },
        "grid": {
            "rows": len(titles),
            "columns": 1,
            "pattern": 'independent',
            'ygap': 0.05,
        },
        'font': {'size': 12},
        "images": [
            {
                "source": logo,
                "xref": "paper",
                "yref": "paper",
                "x": 1,
                "y": 1.008,
                "sizex": .27,
                "sizey": .27,
                "xanchor": "right",
                "yanchor": "bottom"
            },
        ],
    }

    try:
        x_range = [date_from, date_to]
    except IndexError:
        x_range = None

    for i, title in enumerate(titles):
        if not title:
            continue

        i = i + 1  # We want 1 based numbering here
        y_axis = f'yaxis{i}'
        x_axis = f'xaxis{i}'

        layout[y_axis] = {
            "zeroline": False,
            'title': title,
            'gridcolor': 'rgba(0,0,0,.3)',
            'showline': True,
            'showgrid': False,
            'linecolor': 'rgba(0,0,0,.5)',
            'mirror': True,
            'ticks': "inside"
        }

        layout[x_axis] = {
            'automargin': True,
            'autorange': False,
            'range': x_range,
            'type': 'date',
            'tickformat': '%m/%d/%y<br>%H:%M',
            'hoverformat': '%m/%d/%Y %H:%M:%S',
            'gridcolor': 'rgba(0,0,0,.3)',
            'showline': True,
            'showgrid': False,
            'mirror': True,
            'linecolor': 'rgba(0,0,0,.5)',
            'ticks': "inside"
        }

        if i != len(titles):  # All but the last one
            layout[x_axis]['matches'] = f'x{len(titles)}'
            layout[x_axis]['showticklabels'] = False

    return layout


@app.route('/api/gen_graph', methods=["POST"])
@compressor.compressed()
def gen_graph_from_web():
    print("API gen_graph request received")
    data = json.loads(flask.request.form['data'])
    print("Pulled data from request")
    layout = json.loads(flask.request.form['layout'])
    print("Pulled layout from request")

    # Fix up images in layout (using a URL doesn't seem to work in my testing)
    static_path = os.path.join(app.static_folder, 'img')

    for img in layout['images']:
        img_name = img['source'].split('/')[-1]
        img_path = os.path.join(static_path, img_name)
        print("Image set with path:", img_path)
        img_file = Image.open(img_path)
        img['source'] = img_file

    # Shift the title over a bit
    layout['title']['x'] = .09
    layout['title']['y'] = .95
    print("Calling gen_graph_image")
    return gen_graph_image(data, layout)


def gen_graph_image(data, layout, fmt = 'pdf', disposition = 'download',
                    width = 768):

    # Change plot types to scatter instead of scattergl. Bit slower, but works
    # properly with PDF output
    print("Fixing up data")
    for plot in data:
        # We want regular plots so they come out good
        if plot['type'].endswith('gl'):
            plot['type'] = plot['type'][:-2]

    print("Fixing up plot title")
    plot_title = layout['title']['text']
    plot_title = plot_title.replace(' ', '_')
    plot_title = plot_title.replace('/', '_')

    args = {'data': data,
            'layout': layout, }

    print("Creating Figure")

    import pickle
    with open('/tmp/createData.pickle', 'wb') as file:
        pickle.dump(args, file)

    fig = go.Figure(args)
    print("Figure Created. Getting Bytes")

    # TEMPORARY DEBUG
    #    filename = f"{uuid.uuid4().hex}.pdf"
    #     fig.write_image(os.path.join('/tmp', filename), width = 600, height = 900,
    #                     scale = 1.75)

    # Since we chose 600 for the "width" parameter of the to_image call
    # Adjust the output size by using scale, rather than changing the
    # width/height of the call. Seems to work better for layout.
    scale = min(width / 750, 22)
    t1 = time.time()
    fig_bytes = fig.to_image(format = fmt, width = 750, height = 1000,
                             scale = scale)
    print("Called fig.to_image in", time.time() - t1)
    print("Bytes created")
    response = flask.make_response(fig_bytes)
    print("Response created from bytes")

    content_type = f'application/pdf' if fmt == 'pdf' else f'image/{fmt}'
    response.headers.set('Content-Type', content_type)
    if disposition == 'download':
        response.headers.set('Content-Disposition', 'attachment',
                             filename = f"{plot_title}.{fmt}")
    else:
        response.headers.set('Content-Disposition', 'inline')
    print("Returning Response")
    return response


@app.route('/map/download', methods=["POST"])
@compressor.compressed()
def gen_map_image():
    # has to be imported at time of use to work with uwsgi
    try:
        import pygmt
    except Exception:
        os.environ['GMT_LIBRARY_PATH'] = '/usr/local/lib'
        import pygmt

    map_bounds = json.loads(flask.request.form['map_bounds'])
    bounds = [map_bounds['west'],
              map_bounds['east'],
              map_bounds['south'],
              map_bounds['north']]

    fig = pygmt.Figure()
    fig.basemap(projection="M10i", region=bounds, frame=('WeSn', 'afg'))

    if bounds[3] > 60:
        parent_dir = os.path.dirname(__file__)
        grid = os.path.join(parent_dir, "alaska_2s.grd")
    else:
        grid = '@srtm_relief_01s'

    fig.grdimage(grid, cmap = 'geo', dpi = 600, shading = True, monochrome = True)
    fig.coast(rivers = 'r/2p,#FFFFFF', water = "#FFFFFF", resolution = "f")

    if not stations:
        stations.fetch()

    station_data = pd.DataFrame.from_dict(stations, orient = "index")

    fig.plot(x = station_data.lng, y = station_data.lat,
             style = "c0.5i",
             color = '73/0/244',
             pen = '2p,white')

    fig.text(x = station_data.lng, y = station_data.lat,
             text = station_data.index.tolist(), font = "12p,Helvetica-Bold,white")

    #   fig.show(method = "external")
    save_file = f'{uuid.uuid4().hex}.pdf'
    file_path = os.path.join('/tmp', save_file)
    fig.savefig(file_path)

    file = open(file_path, 'rb')
    file_data = file.read()
    file.close()
    os.remove(file_path)

    response = flask.make_response(file_data)
    response.headers.set('Content-Type', 'application/pdf')
    response.headers.set('Content-Disposition', 'attachment',
                         filename = "MapImage.pdf")

    return response


def _get_scale_parameters(scale_len, bounds):
    left, right, bottom, top = bounds
    height = top - bottom
    width = right - left

    date_top = top - .005 * height
    text_top = top - .02 * height
    vector_top = top - .04 * height
    desired_left = left + .01 * width

    background_top = top
    background_bottom = top - .05 * height
    background_left = left

    # Scale_len in in meters, convert to degrees longitude
    deg_len = scale_len / (111320 * numpy.cos(vector_top * (numpy.pi / 180)))
    desired_right = desired_left + deg_len

    background_right = left + deg_len + 0.06 * width

    x = [numpy.float32(desired_left), numpy.float32(desired_right)]
    y = [numpy.float32(vector_top), numpy.float32(vector_top)]
    return {'background': [background_left, background_bottom,
                           background_right, background_top],
            'x': x,
            'y': y,
            'text': text_top,
            'dt': date_top, }


@app.route('/list_stations')
def list_stations():
    max_age = int(flask.request.args.get('age', -1))
    if max_age < 0:
        max_age = 10

    stns = utils.load_stations(max_age)
    return flask.jsonify(stns)


def parse_req_args():
    station = flask.request.args['station']
    channel = flask.request.args['channel']
    date_to = flask.request.args.get(
        'dateTo',
        datetime.now(tz = timezone.utc).replace(hour = 23,
                                                minute = 59,
                                                second = 59,
                                                microsecond = 9999)
    )
    try:
        date_to = parse(date_to)
        date_to = date_to.replace(tzinfo = timezone.utc, hour = 23,
                                  minute = 59, second = 59, microsecond = 9999)
    except TypeError:
        pass

    date_from = flask.request.args.get('dateFrom', date_to-timedelta(days = 7))
    try:
        date_from = parse(date_from)
        date_from = date_from.replace(tzinfo = timezone.utc, hour = 0,
                                      minute = 0, second = 0, microsecond = 0)
    except TypeError:
        pass

    factor = flask.request.args.get('factor', "auto")
    try:
        factor = int(factor)
    except ValueError:
        if factor != 'auto':
            return flask.abort(422)

    return {
        'station': station,
        'channel': channel,
        'date_from': date_from,
        'date_to': date_to,
        'factor': factor,
    }

@app.route('/get_full_data')
@compressor.compressed()
def get_full_data():
    args = parse_req_args()
    app.logger.info(f"Got request for full data download: {args}")

    date_from = args['date_from']
    date_to = args['date_to']

    from_str = date_from.strftime('%Y%m%dT%H%M%S')
    to_str = date_to.strftime('%Y%m%dT%H%M%S')

    filename = f"{args['station']}-{args['channel']}-{from_str}-{to_str}.csv"

    shannon_column = 'entropy'
    shannon_channel = args['channel'][-1]
    if shannon_channel != 'Z':
        shannon_column += f"_{shannon_channel}"

    SQL = f"""
SELECT
    to_char(coalesce(datetime, time) AT TIME ZONE 'UTC','YYYY-MM-DD"T"HH24:MI:SS"Z"') as date,
    freq_max10,
    sd_freq_max10,
    rsam,
    entropy
FROM data d1
FULL OUTER JOIN
	(SELECT
            date_trunc('seconds',time) as time,
            {shannon_column}
	FROM shannon_entropy
	WHERE
            time>=%(date_from)s
            AND time<=%(date_to)s
            AND {shannon_column} > 0
            AND {shannon_column} != 'NaN'
            AND station=%(staid)s
) se
ON se.time=date_trunc('seconds', d1.datetime)
WHERE
    d1.station=%(staid)s
    AND channel=%(channel)s
    AND freq_max10!='NaN'
    AND sd_freq_max10!='NaN'
    AND rsam!='NaN'
    AND datetime>=%(date_from)s
    AND datetime<=%(date_to)s
ORDER BY 1
"""

    def generate():
        # Output the header immediately so we go straight to downloading.
        yield ",".join( ('date', 'freq_max10', 'sd_freq_max10', 'rsam', 'entropy', '\n') )
        app.logger.info("Headers sent. Running data query")
        with utils.db_cursor() as cursor:
            # Look up the ID of the station first, so postgresql can look only at the proper subtable.
            cursor.execute('SELECT id FROM stations WHERE name=%(station)s', args)
            sta_id = cursor.fetchone()
            if not sta_id:
                return flask.abort(404) # Really shouldn't happen...

            args['staid'] = sta_id[0]

            _t1 = time.time()
            cursor.execute(SQL, args)
            app.logger.info(f"Ran query in {time.time()-_t1}" )

            #Output the records, formatted as CSV
            for row in cursor:
                yield ",".join(str(x) if x is not None else '' for x in row) + "\n"

            app.logger.info(f"Completed CSV in {time.time() - _t1}")

    output = flask.Response(
        generate(),
        mimetype = 'text/csv',
        content_type = 'text/csv',
        headers = {'Content-Disposition': f"attachment; filename={filename}",}
    )
    return output

@app.route('/get_graph_data')
@compressor.compressed()
def get_graph_data(as_json=True, station=None, channel = None,
                   date_from=None, date_to=None, factor = "auto"):

    if station is None:
        args = parse_req_args()
        station = args['station']
        channel = args['channel']
        date_from = args['date_from']
        date_to = args['date_to']
        factor = args['factor']

    data = load_db_data(station, channel,
                        date_from=date_from,
                        date_to=date_to,
                        factor=factor)

    resp_data = {'factor': factor,
                 'data': data}

    if as_json:
        str_data = ujson.dumps(resp_data)
        return flask.Response(response=str_data, status=200,
                              mimetype="application/json")
    else:
        return data


PERCENT_LOOKUP = {
    .1: '9000=0',
    1: '90=0',  # Keep every 90th row
    5: '20=0',  # keep every 20th row
    25: '4=0',  # keep every fourth row
    50: '2=0',  # keep every other row
    75: '4>0'  # skip every fourth row
}


def load_db_data(station, channel,
                 date_from=None, date_to=None,
                 factor = 100):

    graph_data = {
        'freq_max10': [],
        'sd_freq_max10': [],
        'rsam': [],
        'dates': [],
        'entropy_dates': [],
        'entropies': [],
        'info': {

        },
    }

    split_table_name = f"data_{station.lower()}_{channel.lower()}"
    rows_desired = 10000
    row_percent = 1

    tablesample_factor = 100
    print(date_to-date_from)
    if date_from and date_to:
        row_percent ="""
        (extract(epoch FROM (%(stop)s-%(start)s))/ -- number of seconds we want
         extract(epoch FROM (max(datetime)-min(datetime))) --number of seconds in table
        )
        """
        if factor == 'auto':
            days = date_to - date_from
            if days > timedelta(days = 365):
                tablesample_factor = .5
            elif days > timedelta(days = 90):
                tablesample_factor = 2

    elif date_from:
        # We want all data to today
        row_percent ="""
            (extract(epoch FROM (now() AT TIME ZONE 'UTC'-%(start)s))/ -- number of seconds we want
              extract(epoch FROM (max(datetime)-min(datetime))) --number of seconds in table
            )
            """
        if factor =='auto' and (datetime.utcnow() - date_from) > timedelta(days = 90):
            tablesample_factor = 2
    elif date_to:
        row_percent ="""
        (extract(epoch FROM (%(stop)s-min(datetime)))/ -- number of seconds we want
         extract(epoch FROM (max(datetime)-min(datetime))) --number of seconds in table
        )
        """
        if factor == 'auto':
            tablesample_factor = 2

    PERCENT_SQL = f"""
    SELECT -- percentage of rows from full table we need to get the desired result count
    (SELECT reltuples*{tablesample_factor / 100} FROM pg_class WHERE relname = '{split_table_name}') /
	(({rows_desired})/ --number of rows we want, times 100 (percentage)
	 LEAST(1,
               {row_percent}
         ) --percentage of the table we are interested in
	)
        FROM {split_table_name}
"""

    print("Using tablesample factor of", tablesample_factor)
    SQL = """
SELECT
    to_char(datetime AT TIME ZONE 'UTC','YYYY-MM-DD"T"HH24:MI:SS"Z"') as text_date,
    freq_max10,
    sd_freq_max10,
    rsam
FROM data"""

    if tablesample_factor < 100:
        SQL += f" TABLESAMPLE SYSTEM({tablesample_factor})"

    SQL += """
WHERE
    station=%(staid)s
    AND channel=%(channel)s
    AND freq_max10!='NaN'
    AND sd_freq_max10!='NaN'
    AND rsam!='NaN'
"""

    shannon_column = 'entropy'
    shannon_channel = channel[-1]
    if shannon_channel != 'Z':
        shannon_column += f"_{shannon_channel}"

    SHANNON_SQL = f"""
    SELECT
        to_char(time AT TIME ZONE 'UTC','YYYY-MM-DD"T"HH24:MI:SSZ') as "entropy_dates",
        {shannon_column} as "entropies"
    FROM shannon_entropy
    WHERE station=%(staid)s
    AND {shannon_column} > 0
    AND {shannon_column} != 'NaN'
    """

    args = {}
    with utils.db_cursor() as cursor:
        cursor.execute("SELECT id FROM stations WHERE name=%s", (station,))
        try:
            station_id = cursor.fetchone()[0]
            args['staid'] = station_id
        except TypeError:
            return graph_data  # Empty set, reference station not found.

        args['channel'] = channel

        cursor.execute(
            """
SELECT
	to_char(mintime AT TIME ZONE 'UTC','YYYY-MM-DD"T"HH24:MI:SSZ'),
	to_char(maxtime AT TIME ZONE 'UTC','YYYY-MM-DD"T"HH24:MI:SSZ')
FROM
(SELECT
	   min(datetime) as mintime,
	   max(datetime) as maxtime
	FROM data
	  WHERE station=%(staid)s
	  AND channel=%(channel)s
) s1;
        """,
            args
        )

        info = cursor.fetchone()
        graph_data['info']['min_date'] = info[0]
        graph_data['info']['max_date'] = info[1]

        if date_from is not None:
            SQL += " AND datetime>=%(start)s"
            SHANNON_SQL += " AND time>=%(start)s"
            args['start'] = date_from
        if date_to is not None:
            SQL += " AND datetime<=%(stop)s"
            SHANNON_SQL += " AND time<=%(stop)s"
            args['stop'] = date_to

        # Get the percentage of the table to work with
        if factor == 'auto':
            cursor.execute(PERCENT_SQL, args)
            factor = cursor.fetchone()[0]
            epoch = f'{int(round(factor))}=0'
        else:
            epoch = PERCENT_LOOKUP.get(factor,'1=0');


        print("Running query with factor", epoch)
        postfix = f" AND epoch%%{epoch}"
        SQL += postfix

        SQL += """
        ORDER BY datetime
        """

        SQL_HEADER = ('dates', 'freq_max10', 'sd_freq_max10',
                      'rsam')

        t1 = time.time()
        cursor.execute(SQL, args)
        if cursor.rowcount != 0:
            # return graph_data  # No data
            results = pd.DataFrame(
                cursor.fetchall(),
                columns = SQL_HEADER).to_dict(orient = "list")
            graph_data.update(results)

        # Get shannon entropy data
        SHANNON_SQL += " ORDER BY time"
        cursor.execute(SHANNON_SQL, args)
        if cursor.rowcount != 0:
            results = pd.DataFrame(
                cursor.fetchall(),
                columns = ("entropy_dates", "entropies")
            ).to_dict(orient = "list")
            graph_data.update(results)


        print("Ran query in", time.time() - t1)
        print("Got", len(graph_data['dates']), "rows in", time.time() - t1, "seconds")
        return graph_data

@app.route('/getdVvData')
def get_dvv_data():
    sta1 = flask.request.args['sta1']
    sta2 = flask.request.args['sta2']
    
    SQL = "SELECT datetime, coh, dvv FROM wct WHERE sta1=%s and sta2=%s ORDER BY datetime;"
    with utils.db_cursor() as cursor:
        
        cursor.execute('SELECT id FROM stations WHERE name in %s', ((sta1, sta2), ) )
        pair = [x[0] for x in cursor]
        
        cursor.execute(SQL, pair)
        dvv_data = pandas.DataFrame(cursor, columns=['date', 'coh', 'dvv'])
        
    dvv_data = dvv_data.set_index(['date'])
    
    # make sure I have data for every half hour, even if NaN
    dvv = dvv_data['dvv'].apply(pandas.Series)
    dvv = dvv.asfreq('30T', fill_value=numpy.NaN)
    dvv_ewm = dvv.ewm(span=30).mean().astype(float)
    dvv_ewm[dvv.isnull()] = numpy.nan
    
    dvv_dates = pandas.Series(dvv_ewm.index).dt.strftime('%Y-%m-%d %H:%M:%S').tolist()
    dvv_ewm = dvv_ewm.T.sort_index()
    dvv_freq = dvv_ewm.index.tolist()
    dvv_values = dvv_ewm.values.tolist()
    
    coh = dvv_data['coh'].apply(pandas.Series)
    coh_ewm = coh.ewm(span=1).mean().astype(float).values
        
    # dvv_data['em1'] = dvv_data['m'] - dvv_data['em']
    # dvv_data['em2'] = dvv_data['m'] + dvv_data['em']
    # dvv_data['date'] = dvv_data['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    # del dvv_data['em']
    
    # result = dvv_data.to_dict('list')
    
    result = {
        'heatX': dvv_dates,
        'heatY': dvv_freq,
        'heatZ': dvv_values,
    }
    str_data = ujson.dumps(result)
    str_data = str_data.replace("NaN", "null")
    return flask.Response(response=str_data, status=200,
                          mimetype="application/json")


@app.route('/getdVvPairs')
def get_dvv_pairs():
    sta = flask.request.args['station']
    SQL = '''SELECT distinct
        (SELECT name FROM stations WHERE id IN (sta1,sta2) AND id!=%s) pair
    FROM wct
    WHERE %s <@ sta_pair
    ORDER BY 1'''
    
    with utils.db_cursor() as cursor:
        cursor.execute('SELECT id FROM stations WHERE name=%s', (sta, ))
        staid = cursor.fetchone()[0]
        
        cursor.execute(SQL, [staid, [staid]])
        pairs = [x[0] for x in cursor]
        
    json_pairs = ujson.dumps(pairs)
    return json_pairs


    
@app.route('/listRegionEntropies')
def list_region_entropies():
    bounds = json.loads(flask.request.args['bounds'])
    SQL = """SELECT
        name,id
    FROM stations
    WHERE location &&
    ST_MakeEnvelope(%(west)s,%(south)s,%(east)s,%(north)s,4326)
    AND EXISTS (SELECT 1
	FROM shannon_entropy
	WHERE station=stations.id
        AND time>now()-'10 years'::interval
	LIMIT 1)
    ORDER BY name
    """
    with utils.db_cursor() as cursor:
        cursor.execute(SQL, bounds)
        stations = cursor.fetchall()

    result = {}
    for station in stations:
        station, station_id = station
        img = f"static/img/entropy/{station}.png"
        result[station] = [img, station_id]

    return flask.jsonify(result)


@app.route('/listVolcEntropies')
def list_volc_entropies():
    volc = flask.request.args['volc']
    SQL = """
    SELECT
        name,
        id
    FROM
    (SELECT
        name,
        id,
        (stations.location <-> (SELECT location FROM volcanoes WHERE site=%(volc)s))/1000 as dist
    FROM
    stations) s1
    WHERE dist<=(SELECT radius FROM volcanoes WHERE site=%(volc)s)
    AND EXISTS
    (SELECT 1
        FROM shannon_entropy
        WHERE station=s1.id
        AND time>now()-'10 years'::interval
        LIMIT 1)
    """
    with utils.db_cursor() as cursor:
        cursor.execute(SQL, {'volc': volc})
        stations = cursor.fetchall()

    result = {}
    for station in stations:
        station, station_id = station
        img = f"static/img/entropy/{station}.png"
        result[station] = [img, station_id]

    return flask.jsonify({'volc': volc, 'stations': result})


@app.route('/listEntropies', methods=["POST"])
def list_entropies():
    volcs = flask.request.form.getlist('volcs[]')
    SQL = """
SELECT
    volcano,
    station,
    staid
FROM	
    (SELECT
            stations.name as station,
            volcanoes.site as volcano,
            stations.id as staid,
            stations.location <-> volcanoes.location as dist,
            (SELECT 1 FROM shannon_entropy WHERE shannon_entropy.station=stations.id LIMIT 1) as exists
    from stations
    CROSS JOIN volcanoes
    WHERE stations.location <-> volcanoes.location <= volcanoes.radius*1000
    AND volcanoes.site = ANY(%s)) s1
WHERE exists=1
ORDER BY volcano, station"""
    with utils.db_cursor() as cursor:
        cursor.execute(SQL, [volcs, ])
        results = pandas.DataFrame(cursor, columns=['volc', 'station', 'staid'])
        
    results['img'] = "static/img/entropy/" + results['station'] + ".png"
    grouped = results.groupby('volc')
    
    output = []
    for name, group in grouped:
        group = group.set_index('station')
        volc_stations = group[['img', 'staid']].to_dict('index')
        output.append({'volc': name,'stations': volc_stations,})
        
    return flask.jsonify(output)

@app.route('/listEventImages')
def list_event_images():
    SQL = """
    SELECT
        array_agg(name),
        array_agg(id),
        site
    FROM
    (SELECT
            name,
            id,
            location
    FROM
    stations) s1
    INNER JOIN (
        SELECT
            location,
            radius,
            site,
            longitude
        FROM volcanoes
    ) s2
    ON (s1.location <-> s2.location)/1000<=s2.radius
    WHERE EXISTS (SELECT 1 FROM last_data WHERE station=s1.id AND lastdata>now()-'10 years'::interval LIMIT 1)
    GROUP BY site,longitude
    ORDER BY CASE WHEN longitude>0 THEN longitude-360 ELSE longitude END DESC
    """

    result = {}
    file_path = os.path.dirname(__file__)
    img_path = 'static/img/events'
    full_path = os.path.join(file_path, img_path)
    with utils.db_cursor() as cursor:
        cursor.execute(SQL)
        for siteinfo in cursor:
            stations, ids, site = siteinfo
            if not site:
                continue

            images = []
            for station, staid in zip(stations, ids):
                img_pattern = f'{station}-?HZ.png'
                # We assume this only returns one, as there should only be one Z channel per station.
                try:
                    file = glob.glob(os.path.join(full_path, img_pattern))[0]
                except IndexError:
                    app.logger.warning(f"No event image found for station {station}")
                    continue

                file_path = os.path.join(img_path, os.path.basename(file))
                images.append((station, file_path, staid))
            result[site] = images

    # flask.jsonify sorts the result, which is incorrect for this usage,
    # so I have to use the "basic" json.dumps instead.
    return json.dumps(result)


@app.route('/getEventData')
def get_event_data():
    site = flask.request.args['volc']
    end = flask.request.args.get('eventEnd', datetime.utcnow())
    begin = flask.request.args.get('eventBegin', end - timedelta(days = 31))

    end = end + timedelta(days = 1)
    end = end.date()
    begin = begin.date()

    SQL = """
WITH volc_info AS (
    SELECT
            location,
            radius
    FROM volcanoes
    WHERE site=%s
)
SELECT
	stations.name,
	ensemble,
	event_begin,
	event_end,
	duration,
	ampl,
	frequency,
	channel
FROM events
INNER JOIN stations ON events.station=stations.id
WHERE (stations.location <-> (SELECT location FROM volc_info))/1000<= (SELECT radius FROM volc_info)
AND event_begin>%s
AND event_begin<%s
ORDER BY stations.name, ensemble;
    """

    args = (site, begin, end)

    filename = f"Events_{site}_{begin.strftime('%Y-%m-%d')}_{end.strftime('%Y-%m-%d')}.csv"
    csv_file = StringIO()
    csv_writer = csv.writer(csv_file)

    with utils.db_cursor() as cursor:
        cursor.execute(SQL, args)
        csv_writer.writerows(cursor)

    output = flask.make_response(csv_file.getvalue())
    output.headers["Content-Disposition"] = f"attachment; filename={filename}"
    output.headers["Content-type"] = "text/csv"
    return output
