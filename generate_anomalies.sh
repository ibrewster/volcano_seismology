#!/bin/bash
cd "${0%/*}/VolcSeismo/web"
source "${0%/*}/bin/activate"
python calc_anomalies.py
rsync --delete --recursive /apps/volcano_seismology/VolcSeismo/web/static/img/anomalies apps.avo.alaska.edu:/volcano_seismology/VolcSeismo/web/static/img/


