# This command works for my production machine. Paths/ports may need 
# to be modified for a different setup.
# Runs UWSGI on container port 5000, mapped to local port 5004, 
# as well as creating a socket in the host /var/run/specweb directory
# Generated plot images are saved to the image /code/specweb/static/plots
# Directory, which is mapped to the local /data/specweb/plots directory for 
# persistant storage
docker run -d --name spectrograms -p 5004:5000 -v /data/specweb/plots:/code/specweb/static/plots -v "/var/run/specweb":"/var/run/specweb" -t seismic_spectrograms