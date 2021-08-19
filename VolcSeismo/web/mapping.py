import csv
import os
import time
import json
import uuid

from datetime import datetime, timedelta, timezone
from io import StringIO

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
    ssa_max = gen_plot_data_dict(data['dates'], data['ssa_max10'], 3)
    sd_ssa_max = gen_plot_data_dict(data['dates'], data['sd_ssa_max10'], 4)

    plot_data = [freq_max, sd_freq_max, ssa_max, sd_ssa_max]

    layout = gen_subgraph_layout(plot_data,
                                 ['Freq Max10 (Hz)', 'SD Freq Max10 (Hz)',
                                  'SSA Max10 (counts)', 'SD SSA Max10 (counts)'],
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
            "size": 4,
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

    fig.grdimage(grid, cmap = 'geo', dpi = 300, shading = True, monochrome = True)
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


@app.route('/get_full_data')
@compressor.compressed()
def get_full_data():
    station = flask.request.args['station']
    channel = flask.request.args['channel']
    date_from = flask.request.args.get('dateFrom')
    date_to = flask.request.args.get('dateTo')
    filename = f"{station}-{channel}-{date_from}-{date_to}.csv"

    data = get_graph_data(False)
    # format as a CSV

    del data['info']

    header = []
    csv_data = []
    for col, val in data.items():
        csv_data.append(val)
        if col == "dates":
            col = "date"
        header.append(col)

    csv_data = numpy.asarray(csv_data).T

    csv_file = StringIO()
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(header)
    csv_writer.writerows(csv_data)

    output = flask.make_response(csv_file.getvalue())
    output.headers["Content-Disposition"] = f"attachment; filename={filename}"
    output.headers["Content-type"] = "text/csv"
    return output


@app.route('/get_graph_data')
@compressor.compressed()
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
