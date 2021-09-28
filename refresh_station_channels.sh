psql -U israel volcano_seismology << EOF
BEGIN;
CREATE TABLE station_channels_new AS (
    SELECT station, array_agg(channel) as channels 
    FROM (
        SELECT station,channel 
        FROM data 
        GROUP BY station,channel) AS s1 
    GROUP BY station);
DROP TABLE station_channels;
ALTER TABLE station_channels_new RENAME TO station_channels;
COMMIT;
EOF