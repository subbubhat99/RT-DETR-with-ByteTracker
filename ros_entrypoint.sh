#!/bin/bash
set -e

rm -rf /usr/bin/python
ln -s /usr/bin/python3 /usr/bin/python
# setup ros environment
source "/opt/ros/$ROS_DISTRO/setup.bash"
exec "$@"