#!/bin/bash
source "${0%/*}/bin/activate"
cd "${0%/*}/VolcSeismo/web"
python calc_anomalies.py
rsync --delete --recursive /apps/volcano_seismology/VolcSeismo/web/static/img/anomalies apps.avo.alaska.edu:/volcano_seismology/VolcSeismo/web/static/img/


