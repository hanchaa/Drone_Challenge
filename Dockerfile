FROM --platform=linux/amd64 rony2:base

SHELL ["/bin/bash", "-c"]

#Setting timezone data
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# set environmet variables for ROS Conn info.
ENV ROS_MASTER_URI "http://192.168.0.20:11311"
ENV ROS_HOST_NAME "192.168.1.40"
ENV REST_ANSWER_URL "http://106.10.49.89:30091/answer"
ENV REST_MISSION_URL "http://106.10.49.89:30090/mission"

WORKDIR /home/agc2022

RUN mkdir dataset
COPY . .
RUN pip install -r requirements.txt
RUN chmod 755 ./task2_audio/receiver.sh
RUN chmod 755 ./run.sh
ENTRYPOINT ["./run.sh"]
