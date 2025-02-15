podman machine start
podman build -t sumo-tools .

# make sure X11 or xQuartz on OSX is running
podman run --privileged --arch amd64 -it --env DISPLAY=host.docker.internal:0 --volume /tmp/.X11-unix:/tmp/.X11-unix sumo-tools
sumo-gui --configuration-file zurich.sumocfg
podman machine stop