--
-- PostgreSQL database cluster dump
--

SET default_transaction_read_only = off;

SET client_encoding = 'SQL_ASCII';
SET standard_conforming_strings = on;

--
-- Roles
--

CREATE ROLE geodesy;
ALTER ROLE geodesy WITH NOSUPERUSER INHERIT NOCREATEROLE NOCREATEDB LOGIN NOREPLICATION NOBYPASSRLS PASSWORD 'SCRAM-SHA-256$4096:MIQZXrKHqGVLbbVMe8RDzQ==$zov+45Tkhihm32Twujm55ZAHr7PRjwB3JMW5+ljfd3U=:FYa8WI4ugx1D0tnNsd7fXpqoXJrPJCYzQJomFVldgpM=';
CREATE ROLE icinga;
ALTER ROLE icinga WITH NOSUPERUSER INHERIT NOCREATEROLE NOCREATEDB LOGIN NOREPLICATION NOBYPASSRLS PASSWORD 'SCRAM-SHA-256$4096:VQXb9IJCoq5TkunC8qxSjw==$OEzxUaMZW5d116GESWRR9qWySkYqBy2PXpokN6HU59Y=:GBq3USaUWJE1qLRkEM4TYbcsUhweTtjGe6siAafSLpk=';
CREATE ROLE israel;
ALTER ROLE israel WITH SUPERUSER INHERIT NOCREATEROLE NOCREATEDB LOGIN NOREPLICATION NOBYPASSRLS PASSWORD 'SCRAM-SHA-256$4096:UB+b2PiO3YJHtseCgdEeoA==$8ko0Xuk4EfFUEO6722XKatOTcUtmRDiS64PGnQgDwFs=:2UsccLI2q6PZGJgphFf4KGnk+U5INjIxUdj9K5DVlrU=';
CREATE ROLE postgres;
ALTER ROLE postgres WITH SUPERUSER INHERIT CREATEROLE CREATEDB LOGIN REPLICATION BYPASSRLS;
CREATE ROLE specgen;
ALTER ROLE specgen WITH NOSUPERUSER INHERIT NOCREATEROLE NOCREATEDB LOGIN NOREPLICATION NOBYPASSRLS PASSWORD 'SCRAM-SHA-256$4096:IosgfBkrMNoustqx0OHGgA==$WBOBt3DwszfueNR3hKpgwJdJFEJrFqRCbTCf7nEZpJI=:70jCo3nTJByIQuJ9MqM2oBcvoyLmnBPr5IaiS6+zBj8=';
CREATE ROLE timescale;
ALTER ROLE timescale WITH NOSUPERUSER INHERIT NOCREATEROLE NOCREATEDB LOGIN NOREPLICATION NOBYPASSRLS;

--
-- User Configurations
--






--
-- Tablespaces
--

CREATE TABLESPACE pool OWNER postgres LOCATION '/pool/psql-17';
GRANT ALL ON TABLESPACE pool TO specgen;


--
-- PostgreSQL database cluster dump complete
--
