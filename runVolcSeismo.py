from VolcSeismo.web import app

if __name__ == "__main__":
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 10
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=False, port=5050, use_reloader=True, host="0.0.0.0")

