#!/bin/bash

psql -U israel volcano_seismology << EOF
BEGIN;
CREATE TABLE station_channels_new AS (
    SELECT station, array_agg(channel) as channels 
    FROM (
        SELECT station,channel 
        FROM part_data 
        GROUP BY station,channel) AS s1 
    GROUP BY station);
GRANT SELECT ON station_channels_new TO specgen;
GRANT SELECT ON station_channels_new TO geodesy;
DROP TABLE station_channels;
ALTER TABLE station_channels_new RENAME TO station_channels;
COMMIT;
-- Refresh the last data view
REFRESH MATERIALIZED VIEW CONCURRENTLY last_data;
EOF