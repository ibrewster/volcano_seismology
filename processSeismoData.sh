#!/bin/bash

/apps/volcano_seismology/bin/python -u /apps/volcano_seismology/processSeismoData.py
rsync -a --stats --delete /apps/volcano_seismology/VolcSeismo/web/static/img/events apps.avo.alaska.edu:/shared/apps/avosmart/VolcSeismo/web/static/img/
rsync -a --stats --delete /apps/volcano_seismology/VolcSeismo/web/static/img/entropy apps.avo.alaska.edu:/shared/apps/avosmart/VolcSeismo/web/static/img/
