FROM --platform=linux/amd64 rony2:base

SHELL ["/bin/bash", "-c"]

#Setting timezone data
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# set environmet variables for ROS Conn info.
ENV ROS_MASTER_URI "http://192.168.0.20:11311"
ENV ROS_HOST_NAME "192.168.1.40"

WORKDIR /home/agc2022

RUN mkdir dataset
COPY ./src .
RUN pip install -r requirements.txt
RUN chmod 755 ./task2_audio/receiver.sh
RUN chmod 755 ./run.sh
ENTRYPOINT ["./run.sh"]
