FROM ghcr.io/eclipse-sumo/sumo:main
ENV SIMULATION_FILES_DIR=/app/simulation
WORKDIR $SIMULATION_FILES_DIR
COPY . $SIMULATION_FILES_DIR

# convert the zurich osm export into sumo
RUN netconvert --osm-files map.osm -o zurich.net.xml

# generate routings from converted osm
RUN python3 /usr/share/sumo/tools/randomTrips.py -n zurich.net.xml -o zurich.rou.xml


CMD ["bash"]
#CMD ["sumo", "--configuration-file", "/app/simulation/hello.sumocfg", "--full-output", "/app/simulation/result.xml"]