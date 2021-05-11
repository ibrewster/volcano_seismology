import flask
import json
import os

app = flask.Flask(__name__)
app.secret_key = 'Correct Horse Battery Staple Secret Key Code'

# Make sure the home directory is set to a reasonable location (i.e. NOT /root)
if os.environ.get('HOME', '/root').startswith('/root'):
    home_dir = os.path.dirname(__file__)
    home_dir = os.path.realpath(os.path.join(home_dir, '..'))
    os.environ['HOME'] = home_dir
    app.logger.error(f"HOME DIR Set To: {os.environ.get('HOME')}")

# make sure /usr/local/bin is in path so orca can be found
if not '/usr/local/bin' in os.environ['PATH']:
    os.environ['PATH'] += os.pathsep + '/usr/local/bin'
    app.logger.error("Added /usr/local/bin to PATH")

# Configure ORCA
import plotly.io as pio
pio.orca.config.use_xvfb = True
pio.orca.config.save()

app.jinja_env.filters['jsonify'] = json.dumps

from . import mapping
