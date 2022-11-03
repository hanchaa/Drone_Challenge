
SERVER_IP=$(ifconfig eth0 | grep "inet " | awk -F'[: ]+' '{ print $3 }')
echo $SERVER_IP
gst-launch-1.0 -v udpsrc uri=udp://$SERVER_IP:5000 \
   caps="application/x-rtp,channels=(int)2,format=(string)S16LE,media=(string)audio,payload=(int)96,clock-rate=(int)16000,encoding-name=(string)L24" \
   ! rtpL24depay ! audioconvert \
   ! wavenc ! filesink location=wavwavwavwav.wav