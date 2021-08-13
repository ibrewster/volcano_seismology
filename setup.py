#!/usr/bin/env python3
import getpass
import os
import subprocess
import sys

if __name__ == "__main__":
    assume_yes = False
    if len(sys.argv) > 1 and sys.argv[1] == '-y':
        assume_yes = True
    print("Welcome to the Volcano Seismology setup script!")
    print("This script will walk you through the steps needed to get up and running.")
    cont = 'y' if assume_yes else 'invalid'
    while cont.lower() not in ['', 'y', 'n']:
        cont = input("Continue? [Y/n]: ")

    if cont.lower() == 'n':
        exit()

    if os.environ.get('MySQL__DB_USER') is None:
        print("Please enter the username for the MySQL database")
        db_user = input("containing volcano metadata:")
        os.environ['MySQL__DB_USER'] = db_user

    if os.environ.get('MySQL__DB_PASSWORD') is None:
        print("Please enter the password for the MySQL database")
        db_pass = getpass.getpass("containing volcano metadata:")
        os.environ['MySQL__DB_PASSWORD'] = db_pass

    print("Installing base requirements")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade",
                           "-r", "requirements.txt"])

    gen_base_conf = 'y' if assume_yes else 'invalid'
    while gen_base_conf.lower() not in ('', 'y', 'n'):
        gen_base_conf = input("Generate new config.ini file? [Y/n]: ")

    if gen_base_conf.lower() != 'n':
        from VolcSeismo.config.gen_config import main as generage_config
        generage_config()
        print("config.ini has been generated.")
        print("Please verify the contents of this file and edit")
        print("To suit your enviroment before continuing.")
        if not assume_yes:
            input("Hit return when ready to continue.")

    gen_station = 'y' if assume_yes else 'invalid'
    while gen_station.lower() not in ('', 'n', 'y'):
        gen_station = input("Generate new station config file? [Y/n]: ")

    if gen_station.lower() != 'n':
        from VolcSeismo.config.gen_station_config import generate_stations
        print("Please verify that the settings and VOLCS list at the top of the")
        print("gen_station_config.py file are as desired")
        if not assume_yes:
            input("Hit return when ready to continue")

        print("Generating station config. This may take a few minutes...")
        generate_stations()
        print("Station config generated")

    print()
    print("Hooks are user-supplied scripts that operate on the retrieved and processed")
    print("data to provide additional functionality. Some hooks may require additional")
    print("modules to be installed in order to function.")
    print()
    print("This step is optional, if a module required for a hook is not installed")
    print("that hook will simply not be run.")
    install_hooks = 'y' if assume_yes else 'invalid'
    while install_hooks.lower() not in ('', 'n', 'y'):
        install_hooks = input("Install required modules for hooks? [Y/n]: ")

    if install_hooks.lower() != 'n':
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade",
                               "-r", "VolcSeismo/hooks/requirements.txt"])

    print()
    print("Your install is set up and ready to go.")
    print()
    print("The web interface can be launched in development mode by running runVolcSeismo.py")
    print("For production use, it is recommended that you instead use a production-")
    print("quality WSGI server behind a web server such as Nginx or Apache")
    print("See https://flask.palletsprojects.com/en/2.0.x/deploying/ for more information")
