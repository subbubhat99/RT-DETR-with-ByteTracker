ARG ROS_DISTRO=humble
ARG FROM_IMAGE=ubuntu:20.04
ARG OVERLAY_WS=/opt/capra/overlay_ws
ARG ROS_SETUP=/opt/ros/$ROS_DISTRO/setup.sh

FROM $FROM_IMAGE AS ros_install

ARG DEBIAN_FRONTEND=noninteractive
RUN apt update 
RUN apt install -y software-properties-common curl gnupg lsb-release 
RUN add-apt-repository universe
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu jammy main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null
RUN rm /var/lib/dpkg/info/libc-bin.*
RUN apt-get clean
RUN apt-get update
RUN apt-get install libc-bin

ARG ROS_DISTRO
ARG OVERLAY_WS
ARG ROS_SETUP
RUN apt-get update -y && apt-get install -q -y --no-install-recommends\
    ccache \
    lcov \
    git \
    net-tools \
    iputils-ping \
    build-essential \
    unzip \
    g++ \
    python3-pip \
    python3-colcon-common-extensions \
    python3-flake8 \
    python3-pytest-cov \
    python3-rosdep \
    python3-setuptools \
    python3-vcstool \
    python-is-python3 \
    ros-$ROS_DISTRO-ros-base \
    ros-$ROS_DISTRO-rosidl-default-generators \
    ros-$ROS_DISTRO-tf2-ros \
    ros-$ROS_DISTRO-tf-transformations \
    ros-$ROS_DISTRO-ament-cmake \
    ros-$ROS_DISTRO-cv-bridge \
    ros-$ROS_DISTRO-vision-opencv \
    libgl1

COPY ./requirements.txt requirements.txt
RUN pip3 install -r requirements.txt


# Install the bash-auto complete
RUN sudo apt-get install bash-completion && sudo apt-get install --reinstall bash-completion
ENV DEBIAN_FRONTEND ""

# Assure the timezone is european => CPH
ENV TZ=Europe/Copenhagen
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && date

FROM ros_install AS dds_vendor_setter
ARG ROS_DISTRO
RUN apt-get update \
  && apt-get install -y -q --no-install-recommends\
  ros-$ROS_DISTRO-rmw-fastrtps-cpp \
  && rm -rf /var/lib/apt/lists/*

ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp

FROM dds_vendor_setter AS developer
ARG ROS_DISTRO
ARG OVERLAY_WS
ARG ROS_SETUP
ARG WORKSPACE

WORKDIR $OVERLAY_WS
RUN apt-get update && apt-get install -q -y --no-install-recommends \
  gdb \
  ssh \
  && rm -rf /var/lib/apt/lists/*

WORKDIR $OVERLAY_WS

RUN . $ROS_SETUP && rosdep init && rosdep update
RUN . $ROS_SETUP && colcon build


FROM dds_vendor_setter AS builder
ARG DEBIAN_FRONTEND=noninteractive

# install overlay dependencies
ARG OVERLAY_WS
WORKDIR $OVERLAY_WS
ARG ROS_DISTRO
ARG ROS_SETUP

COPY ./ src/RT-DETR-with-ByteTracker
RUN . $ROS_SETUP && colcon build

FROM builder as runner

ARG ROS_SETUP
ARG WORKSPACE
ARG OVERLAY_WS
WORKDIR $OVERLAY_WS

COPY --from=developer /opt/capra/overlay_ws/install /opt/capra/overlay_ws/install
#COPY --from=developer /opt/capra/overlay_ws/capra_ros /opt/capra/overlay_ws/capra_ros

COPY ros_entrypoint.sh /ros_entrypoint.sh
RUN sudo sed --in-place \
  "s|^source .*|source '$OVERLAY_WS/install/setup.bash'|" \
  /ros_entrypoint.sh

ENTRYPOINT [ "/ros_entrypoint.sh" ]

# This container should always be ran from the docker-compose file, as it is used to either record or play
CMD ["python", "rt-detr-track-inference.py"]