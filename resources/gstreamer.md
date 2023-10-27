# Headless Mode

The output of the headless mode can be viewed without a second CyberEther instance by using an assortment of GStreamer commands. The following commands are examples of how to view and record the output of the headless mode. These are not well maintained and are just here as a reference.

### Displaying the named pipe output in a window. 

```
gst-launch-1.0 filesrc location=/tmp/cyberether ! \
    queue ! \
    video/x-raw,format=BGRA,width=1920,height=1080,framerate=60/1 ! \
    rawvideoparse use-sink-caps=true ! \
    queue ! \
    glimagesink sync=false
```

### Recording the named pipe output to a file as H265 video.

```
gst-launch-1.0 filesrc location=/tmp/cyberether ! \
    video/x-raw,format=BGRA,width=1920,height=1080,framerate=60/1 ! \
    rawvideoparse use-sink-caps=true ! \
    queue ! \
    videoconvert ! \
    x265enc bitrate=100000 ! \
    h265parse ! \
    mp4mux ! \
    filesink location=output.mp4
```

### Recording the named pipe output to a file as a lossless FFV1 video.

```
gst-launch-1.0 filesrc location=/tmp/cyberether ! \
    queue ! \
    video/x-raw,format=BGRA,width=1920,height=1080,framerate=60/1 ! \
    rawvideoparse use-sink-caps=true ! \
    queue ! \
    videoconvert ! \
    avenc_ffv1 ! \
    matroskamux ! \
    filesink location=output.mkv
```

### Streaming the named pipe output over TCP as a lossless FFV1 video.

```
gst-launch-1.0 multifilesrc location=/tmp/cyberether ! \
    video/x-raw,format=BGRA,width=1920,height=1080,framerate=60/1 ! \
    rawvideoparse use-sink-caps=true ! \
    queue ! \
    videoconvert ! \
    avenc_ffv1 ! \
    matroskamux ! \
    tcpserversink host=0.0.0.0 port=8881
```

```
gst-launch-1.0 tcpclientsrc host=127.0.0.1 port=5000 ! \
    matroskademux ! \
    avdec_ffv1 ! \
    videoconvert ! \
    glimagesink sync=false
```