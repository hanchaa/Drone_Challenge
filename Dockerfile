FROM --platform=linux/amd64 rony2:base

# Change the default shell to Bash
SHELL [ "/bin/bash" , "-c" ]

#Setting timezone data
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# set environmet variables for ROS Conn info.
ENV ROS_MASTER_URI "http://115.94.141.61:11311"
ENV ROS_HOST_NAME "10.41.90.220"

WORKDIR /home/agc2022

RUN mkdir dataset
COPY ./src .
CMD ["/bin/bash"]
