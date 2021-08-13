# This command works for my production machine. Paths/ports may need 
# to be modified for a different setup.
# Runs UWSGI on container port 5000, mapped to local port 5005, 
# as well as creating a socket in the host /var/run/volcseismo directory

docker run -d --name volcseismo -p 5005:5000  -v "/var/run/volcseismo":"/var/run/volcseismo" -t volcseismo