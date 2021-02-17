# Eyedroid
Eyedroid is a program for detecting objects or class of objects. Detection is done on  images, videos, live webcam or a specified url stream. You can specify what objects you want to detect, and only those will be detected from the stream. When the searched object is found, the program notifies the user with a popup message.

## Platform support
This program is suitable for powerless devices (raspberry pi and micro computer), it uses YOLO and tensorflow lite in order to be more faster.

## Download eyedroid
Clone the github repository:
```
git clone https://github.com/includecode/eyedroid.git
```

# Installation
We'll install TensorFlow, OpenCV, and all the dependencies needed for both packages. OpenCV is not needed to run TensorFlow Lite, but the object detection scripts in this repository uses it to grab images and draw detection results on them. All the tools needed to run this program successfuly are specified inside **production/TensorFlowLite/get_pi_requirements.sh**. Browse to the folder and type:
```
chmod +x get_pi_requirements.sh

./get_pi_requirements.sh
```

# Usage
## Basic usage
Once all the librairies are installed, browse to the **production/TensorFlowLite/**. We are ready to go:
 - Live webcam detection
 run the following command:
 ```
python3 TFLite_detection_webcam.py --modeldir=yolov4
 ```
 - Video stream
 To run the script to detect images in a video stream (e.g. a remote security camera), issue:
 ```
 python3 TFLite_detection_stream.py --modeldir=yolov4 --streamurl="http://ipaddress:port/stream/video.mjpeg"
 ```

- Detect on an image
 ```
python3 TFLite_detection_image.py --modeldir=yolov4 --image=test1.jpg
 ```

 - Detect from video
 python TFLite_detection_video.py --modeldir=yolov4 --video='test.mp4'

 ## Advanced usage
 For all these 4 detection scripts, you can specify exactly, what objects you want to detect. Write down the objects inside **production/TensorFlowLite/wantedobjects.txt**, each object on a single line and rerun the program. We a new object matches your list, you get a notification.


 # Credits
  - [Training objects]
  (https://www.youtube.com/watch?v=mKAEGSxwOAY)

  - [Basic Script]
  (https://github.com/EdjeElectronics)