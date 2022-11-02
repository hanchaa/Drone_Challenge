FROM osrf/ros:melodic-desktop-full

# Change the default shell to Bash
SHELL [ "/bin/bash" , "-c" ]

#Setting timezone data
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
# Install Git and catkin tools and so on...
RUN apt-get update &&  \ 
    apt-get install -y \
	apt-utils \
	vim \
	sudo \
	tzdata \
	python3-pip \
	python-catkin-tools \
	ros-melodic-mavros \
        build-essential

# set environmet variables for ROS Conn info.
ENV ROS_MASTER_URI "http://115.94.141.61:11311"
ENV ROS_HOST_NAME "10.41.90.220"
ENV HOME=/home

RUN mkdir -p ${HOME}/agc2022/datsaet
WORKDIR ${HOME}/agc2022

COPY ./src .

RUN chmod a+x ./ros_entrypoint.sh
RUN chmod a+x ./user_command.sh
RUN pip3 install scikit-build
RUN pip3 install -r requirements.txt

ENTRYPOINT [ "./ros_entrypoint.sh" ]
CMD ["/bin/bash"]
