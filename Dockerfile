FROM ghcr.io/eclipse-sumo/sumo:main
ENV SIMULATION_FILES_DIR=/app/simulation
WORKDIR $SIMULATION_FILES_DIR
COPY . $SIMULATION_FILES_DIR

# convert the zurich osm export into sumo
RUN netconvert --osm-files map.osm -o zurich.net.xml

# generate random vehicle routes traffic on the map and export it to a SUMO route
RUN python3 /usr/share/sumo/tools/randomTrips.py -n zurich.net.xml -o zurich.rou.xml

CMD ["bash"]
#CMD sumo --configuration-file zurich.sumocfg --full-output result.xml