import os
import math
import time
import json
import uuid

from datetime import datetime, timedelta, timezone

from . import app
from . import utils
from .utils import stations

from VolcSeismo.config import locations

import ujson
import json
import numpy
import flask
import flask.helpers

from dateutil.parser import parse
import pandas as pd
import plotly.graph_objects as go
from PIL import Image


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
    if is_tilt:
        title += ' Tilt '
        data = _get_tilt_data(date_from, date_to, station, raw=True)

        if date_from is None:
            date_from = parse(data['search_start'] + "Z")

        x_trace = gen_tilt_data_dict(data['dates'], data['x_tilt'], 'red', 2)
        y_trace = gen_tilt_data_dict(data['dates'], data['y_tilt'], 'blue', 3)
        temp_trace = gen_tilt_data_dict(data['dates'], data['temps'], 'black', 4)

        date_span = (date_to - date_from)

        if date_span.total_seconds() < 60 * 60 * 24:
            # One day (or less)
            line_dict = {'width', .5}
            for trace in (x_trace, y_trace, temp_trace):
                trace['mode'] = 'lines'
                trace['line'] = line_dict

        polar_graphs = makePolarTrace(data)
        r_max = scalePolarPlot(x_trace['x'], polar_graphs[0], date_from, date_to)

        plot_data = [x_trace, y_trace, temp_trace]

        layout = gen_subgraph_layout(plot_data,
                                     ['', 'X (Microradians)', 'Y (Microradians)', 'Degrees C'],
                                     date_from, date_to)

        plot_data += list(polar_graphs)
        layout = makePolarLayout(layout, data, station)

        if r_max is not None:
            layout['polar']['radialaxis'] = {'range': [0, math.ceil(r_max)]}

        # Change the tick style depending on time span
        minute_span = date_span.total_seconds() / 60
        if minute_span < 5:
            tickformat = "%H:%M:%S"
        elif minute_span <= 36 * 60:  # 36 hours
            tickformat = "%m/%d %H:%M"
        else:
            tickformat = "%Y-%m-%d"

        x_range = [date_from, date_to]
        for axis in ['xaxis2', 'xaxis3', 'xaxis4']:
            layout[axis]['tickformat'] = tickformat
            layout[axis]['range'] = x_range

    else:
        title += ' - '
        data = get_graph_data(False, station=station, baseline_station=baseline,
                              date_from=date_from, date_to=date_to, new_data = new_data)

        if not data['dates']:
            return "no data found", 404

        if date_from is None:
            date_from = parse(data['dates'][0] + "T00:00:00Z")

        ew_trace = gen_plot_data_dict(data['dates'], data['EW'], data['EWE'], data['rapid'])
        ns_trace = gen_plot_data_dict(data['dates'], data['NS'], data['NSE'], data['rapid'], 2)
        ud_trace = gen_plot_data_dict(data['dates'], data['UD'], data['UDE'], data['rapid'], 3)

        plot_data = [ew_trace, ns_trace, ud_trace]
        layout = gen_subgraph_layout(plot_data,
                                     ['EAST (cm)', 'NORTH (cm)', 'VERTICAL (cm)'],
                                     date_from, date_to)
        layout['annotations'] = [{
            "xref": 'paper',
            "yref": 'paper',
            "x": 0.004,
            "xanchor": 'left',
            "y": 1.00,
            "yanchor": 'bottom',
            "text": f"Baseline Station: {baseline}",
            "showarrow": False,
            "font": {"size": 11}
        }]

    title += f'{date_from.strftime("%Y-%m-%d")} to {date_to.strftime("%Y-%m-%d")}'
    layout['title'] = {'text': title,
                       'x': .115,
                       'y': .939,
                       'xanchor': 'left',
                       'yanchor': 'bottom',
                       'font': {
                           'size': 16,
                       }, }

    return gen_graph_image(is_tilt, plot_data, layout, fmt, 'inline',
                           width = width)


def gen_plot_data_dict(x, y, error, color, idx = None):
    trace = {
        "x": x,
        "y": y,
        "error_y": {
            "type": 'data',
            "array": error,
            "visible": True,
            "color": 'rgb(100,50,50)'
        },
        "type": 'scatter',
        "mode": 'lines+markers',
        "marker": {
            "size": 4,
            "cmin": 0,
            "cmax": 1,
            "colorscale": [
                [0, 'rgb(55,128,256)'],
                [1, 'rgb(256,128,55)']
            ],
            "color": color
        },
        "line": {
            "color": 'rgba(55,128,191,.25)'
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
    UAF_GI_LOGO_PATH = os.path.join(script_path, 'static/img/uaf_gi_logo.png')
    uaf_gi_logo = Image.open(UAF_GI_LOGO_PATH)

    AVO_LOGO_PATH = os.path.join(script_path, 'static/img/avo_logo.png')
    avo_logo = Image.open(AVO_LOGO_PATH)

    layout = {
        "paper_bgcolor": 'rgba(255,255,255,1)',
        "plot_bgcolor": 'rgba(255,255,255,1)',
        "showlegend": False,
        "margin": {
            "l": 50,
            "r": 25,
            "b": 25,
            "t": 63,
            "pad": 0
        },
        "grid": {
            "rows": 3,
            "columns": 1,
            "pattern": 'independent',
            'ygap': 0.05,
        },
        'font': {'size': 12},
        "images": [
            {
                "source": uaf_gi_logo,
                "xref": "paper",
                "yref": "paper",
                "x": 1,
                "y": 1.008,
                "sizex": .27,
                "sizey": .27,
                "xanchor": "right",
                "yanchor": "bottom"
            },
            {
                "source": avo_logo,
                "xref": "paper",
                "x": .72,
                "y": 1.008,
                "sizex": .1062,
                "sizey": .1062,
                "xanchor": "right",
                "yanchor": "bottom"
            }],
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

        layout[y_axis] = {"zeroline": False,
                          'title': title,
                          'gridcolor': 'rgba(0,0,0,.3)',
                          'showline': True,
                          'showgrid': False,
                          'linecolor': 'rgba(0,0,0,.5)',
                          'mirror': True,
                          'ticks': "inside"}

        layout[x_axis] = {
            'automargin': True,
            'autorange': False,
            'range': x_range,
            'type': 'date',
            'tickformat': '%Y.%m.%d',
            'hoverformat': '%m/%d/%Y',
            'gridcolor': 'rgba(0,0,0,.3)',
            'showline': True,
            'showgrid': False,
            'mirror': True,
            'linecolor': 'rgba(0,0,0,.5)',
            'ticks': "inside"
        }

        if i != len(titles):
            layout[x_axis]['matches'] = f'x{len(titles)}'
            layout[x_axis]['showticklabels'] = False

    return layout


@app.route('/map/gen_graph', methods=["POST"])
def gen_graph_from_web():
    is_tilt = flask.request.form['is_tilt']
    # Convert to a real boolean
    is_tilt = True if is_tilt == 'true' or is_tilt == '1' else False

    data = json.loads(flask.request.form['data'])
    layout = json.loads(flask.request.form['layout'])

    # Fix up images in layout (using a URL doesn't seem to work in my testing)
    static_path = os.path.join(app.static_folder, 'img')

    for img in layout['images']:
        img_name = img['source'].split('/')[-1]
        img_path = os.path.join(static_path, img_name)
        img_file = Image.open(img_path)
        img['source'] = img_file

    # Shift the title over a bit
    layout['title']['x'] = .09
    layout['title']['y'] = .92

    return gen_graph_image(is_tilt, data, layout)


def gen_graph_image(is_tilt, data, layout, fmt = 'pdf', disposition = 'download',
                    width = 900):

    # Change plot types to scatter instead of scattergl. Bit slower, but works
    # properly with PDF output
    for plot in data:
        # We want regular plots so they come out good
        if plot['type'].endswith('gl'):
            plot['type'] = plot['type'][:-2]

        if 'marker' in plot and 'colorscale' in plot['marker']:
            cs = plot['marker']['colorscale']
            for entry in cs:
                # The first parameter comes in as a string, but needs to be a
                # number
                entry[0] = float(entry[0])

    plot_title = layout['title']['text']
    plot_title = plot_title.replace(' ', '_')
    plot_title = plot_title.replace('/', '_')

    gen_height = 800 if is_tilt else 700

    args = {'data': data,
            'layout': layout, }

    fig = go.Figure(args)

    # TEMPORARY DEBUG
    #    filename = f"{uuid.uuid4().hex}.pdf"
    #     fig.write_image(os.path.join('/tmp', filename), width = 600, height = 900,
    #                     scale = 1.75)

    # Since we chose 600 for the "width" parameter of the to_image call
    scale = min(width / 600, 22)
    fig_bytes = fig.to_image(format = fmt, width = 600, height = gen_height,
                             scale = scale)
    response = flask.make_response(fig_bytes)
    content_type = f'application/pdf' if fmt == 'pdf' else f'image/{fmt}'
    response.headers.set('Content-Type', content_type)
    if disposition == 'download':
        response.headers.set('Content-Disposition', 'attachment',
                             filename = f"{plot_title}.pdf")
    else:
        response.headers.set('Content-Disposition', 'inline')

    return response


@app.route('/map/download', methods=["POST"])
def gen_map_image():
    # has to be imported at time of use to work with uwsgi
    try:
        import pygmt
    except Exception:
        os.environ['GMT_LIBRARY_PATH'] = '/usr/local/lib'
        import pygmt

    vector_args = {}
    for arg in ['baseline', 'date_from', 'date_to', 'scale', 'station',
                'zoom', 'quakes']:
        vector_args[arg] = flask.request.form[arg]

    map_bounds = json.loads(flask.request.form['map_bounds'])
    bounds = [map_bounds['west'],
              map_bounds['east'],
              map_bounds['south'],
              map_bounds['north']]
    date_from = parse(vector_args['date_from']).date()
    date_to = parse(vector_args['date_to']).date()

    fig = pygmt.Figure()
    fig.basemap(projection="M16i", region=bounds, frame=('WeSn', 'afg'))

    parent_dir = os.path.dirname(__file__)
    grid = os.path.join(parent_dir, "alaska_2s.grd")

    fig.grdimage(grid, C = 'geo', E = 300, I = True, M = True)
    fig.coast(I = 'r/2p,#FFFFFF', water = "#FFFFFF", resolution = "f")

    # Plot the earthquakes (if any)
    quake_data = json.loads(vector_args['quakes'])
    quake_df = pd.DataFrame.from_dict(quake_data)

    if quake_df.size > 0:
        quake_df['x'] = quake_df.apply(lambda x: x.location['lng'], axis = 1)
        quake_df['y'] = quake_df.apply(lambda x: x.location['lat'], axis = 1)
        quake_df['sizes'] = quake_df.apply(lambda x: .25 * math.exp(0.5 * x.magnitude),
                                           axis = 1)

        fig.plot(x = quake_df.x, y = quake_df.y,
                 sizes = quake_df.sizes,
                 style = "cc", color = "white", pen = "black")

    # Plot the vectors
    vector_data = _get_vector_data(vector_args['station'],
                                   vector_args['baseline'], date_from,
                                   date_to, int(vector_args['zoom']),
                                   int(vector_args['scale']))

    xy_vectors = pd.DataFrame.from_dict([{'x': numpy.array((x[0]['lng'], x[1]['lng']),
                                                           dtype = numpy.float32),
                                          'y': numpy.array((x[0]['lat'], x[1]['lat']),
                                                           dtype = numpy.float32),
                                          }
                                         for x in vector_data['xy']
                                         ])
    z_vectors = pd.DataFrame.from_dict([{'x': numpy.array((x[0]['lng'], x[1]['lng']),
                                                          dtype = numpy.float32),
                                         'y': numpy.array((x[0]['lat'], x[1]['lat']),
                                                          dtype = numpy.float32),
                                         }
                                        for x in vector_data['z']
                                        ])

    for idx, row in xy_vectors.iterrows():
        fig.plot(x = row.x, y = row.y, pen = '2.5p,51/0/255+ve0.12i+g51/0/255+a50')

    for idx, row in z_vectors.iterrows():
        fig.plot(x = row.x, y = row.y, pen = '2.5p,"#0CDA3B')

    station_data = pd.DataFrame.from_dict(stations, orient = "index")

    # Figure out circle color and pen color
    station_data['color'] = station_data.apply(lambda row: 0 if row.type == "continuous" else 1,
                                               axis = 1)

    # Split the data frame into two so we can vary the pen color
    # seperately from the fill color
    sdata_red = station_data[station_data.has_tilt]
    sdata_white = station_data[~station_data.has_tilt]
    fig.plot(x = sdata_red.lng, y = sdata_red.lat, style = "c0.5i",
             color = sdata_red.color, cmap = "73/0/244,12/218/59",
             pen = '2p,red')

    fig.plot(x = sdata_white.lng, y = sdata_white.lat, style = "c0.5i",
             color = sdata_white.color, cmap = "73/0/244,12/218/59",
             pen = '2p,white')

    fig.text(x = station_data.lng, y = station_data.lat,
             text = station_data.index.tolist(), font = "12p,Helvetica-Bold,white")

    scale_params = _get_scale_parameters(vector_data['scale'], bounds)

    # Plot the scale bar background
    fig.plot(data = numpy.array([scale_params['background']]), style = 'r+s',
             G = "white@75")

    # And the scale bar itself
    fig.plot(x = scale_params['x'], y = scale_params['y'],
             pen = '2.5p,51/0/255+ve0.12i+g51/0/255+a50')

    # some labels
    date_range = f"{date_from.strftime('%m/%d/%y')}-{date_to.strftime('%m/%d/%y')}"
    fig.text(x = scale_params['x'][0], y = scale_params['dt'],
             text = date_range, font = "14p,Helvetica", justify = "TL")

    fig.text(x = scale_params['x'][0], y = scale_params['text'],
             text = '1cm/year', font = "14p,Helvetica", justify = "TL")

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


@app.route('/get_graph_data')
def get_graph_data(as_json=True, station=None, channel = None,
                   date_from=None, date_to=None, factor = "auto"):

    if station is None:
        station = flask.request.args['station']
        channel = flask.request.args['channel']
        factor = flask.request.args.get('factor', "auto")

        try:
            date_from = parse(flask.request.args.get('dateFrom'))
            date_to = parse(flask.request.args.get('dateTo'))
            date_from = date_from.replace(tzinfo = timezone.utc, hour = 0,
                                          minute = 0, second = 0, microsecond = 0)
            date_to = date_to.replace(tzinfo = timezone.utc, hour = 23,
                                      minute = 59, second = 59, microsecond = 9999)
        except (TypeError, ValueError):
            date_to = datetime.now(tz = timezone.utc).replace(hour = 23,
                                                              minute = 59,
                                                              second = 59,
                                                              microsecond = 9999)
            date_from = date_to - timedelta(days = 7)

        if factor == "auto":
            span = (date_to - date_from).total_seconds()
            if span > 2592000:  # One month
                factor = 1
            elif span > 1209600:  # two weeks
                factor = 5
            elif span > 604800:  # one week
                factor = 25
            elif span > 172800:  # two days
                factor = 50
            else:
                factor = 100

        else:
            try:
                factor = int(factor)
            except ValueError:
                return flask.abort(422)

    data = load_db_data(station, channel,
                        date_from=date_from,
                        date_to=date_to,
                        factor = factor)

    if as_json:
        str_data = ujson.dumps(data)
        return flask.Response(response=str_data, status=200,
                              mimetype="application/json")
    else:
        return data


PERCENT_LOOKUP = {
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
        'ssa_max10': [],
        'sd_ssa_max10': [],
        'dates': [],
        'info': {

        },
    }

    SQL = f"""
    SELECT
        to_char(datetime AT TIME ZONE 'UTC','YYYY-MM-DD"T"HH24:MI:SS"Z"') as text_date,
        freq_max10,
        sd_freq_max10,
        ssa_max10,
        sd_ssa_max10
    FROM
        data
    WHERE station=%s
        AND channel=%s
    """

    args = []
    with utils.db_cursor() as cursor:
        cursor.execute("SELECT id FROM stations WHERE name=%s", (station,))
        try:
            station_id = cursor.fetchone()[0]
            args.append(station_id)
        except TypeError:
            return graph_data  # Empty set, reference station not found.

        args.append(channel)

        cursor.execute("""
SELECT
	to_char(mintime AT TIME ZONE 'UTC','YYYY-MM-DD"T"HH24:MI:SSZ'),
	to_char(maxtime AT TIME ZONE 'UTC','YYYY-MM-DD"T"HH24:MI:SSZ')
FROM
(SELECT
	   min(datetime) as mintime,
	   max(datetime) as maxtime
	FROM data
	  WHERE station=%s
	  AND channel=%s
) s1;
        """,
                       args)
        info = cursor.fetchone()
        graph_data['info']['min_date'] = info[0]
        graph_data['info']['max_date'] = info[1]

        if date_from is not None:
            SQL += " AND datetime>=%s"
            args.append(date_from)
        if date_to is not None:
            SQL += " AND datetime<=%s"
            args.append(date_to)

        if factor != 100:
            print("Running query with factor", factor)
            postfix = f" AND epoch%%{PERCENT_LOOKUP.get(factor,'1=0')}"
            SQL += postfix

        SQL += """
        ORDER BY datetime
        """

        SQL_HEADER = ('dates', 'freq_max10', 'sd_freq_max10',
                      'ssa_max10', 'sd_ssa_max10')

        t1 = time.time()
        cursor.execute(SQL, args)
        if cursor.rowcount == 0:
            return graph_data  # No data
        print("Ran query in", time.time() - t1)
        results = numpy.asarray(cursor.fetchall()).T
        t3 = time.time()
        print("Got results in", t3 - t1)
        for idx, key in enumerate(SQL_HEADER):
            graph_data[key] = results[idx].tolist()
        print("processed results in", time.time() - t3)
        print("Got", len(graph_data['dates']), "rows in", time.time() - t1, "seconds")
        return graph_data
