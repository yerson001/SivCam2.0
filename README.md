# Video surveillance using Raspberry PI or USB Camera and Flask

## Author 
Ruben Cardenes, PhD

## Description

This project starts a web server application with Flask that streams video from a camera. 
 It uses motion detection, shows the detected moving objects with red squares, and stores the frames with motion on disk with minimum time separation of 2 seconds. 
The web application allows to see the streaming video a summary of the moving pictures and creates a video summary of the current day.
 
 ## Project features 

- [x] Raspberry PI camera support
- [x] USB Camera support 
- [x] Multi-object motion detection for picture saving 
- [x] Web streaming 
- [x] Direct streaming scripts  
- [x] Picture gallery summary
- [x] Video gallery summary 
- [x] Video summary creation from saved images 
- [x] People detection with AI model
- [x] Access from outside instructions  

![image](example.png)

## Requirements

Python 3.6 

```
Flask==1.1.2
imutils==0.5.3
numpy==1.18.2
opencv-python>=3.4.6.27
tqdm==4.45.0
PyYAML==5.3.1
tensorflow-gpu==1.15.2
```

## Instructions 

1. Download the code:
```
git clone https://github.com/rubencardenes/survey_raspberry_pi.git
```

2. Install python requirements (preferably in a virtual environment) 
```
pip3 install -r requirements.txt 
```

For Raspberry PI, follow the instructions here for tensorflow installation:
```
https://qengineering.eu/install-tensorflow-1.15.2-on-raspberry-pi-4.html
```
In the current versions or Raspbian opencv-python > 4.x gives an error in the atomic library. 
In this code I recommend therefore opencv 3.4.6.x

3. Folder structure 

Make sure that you have a folder called "static" at the same level as the scripts. 
Inside static, you also need to have the folders "images" and "video_summary"
 
4. Start the script (by default on localhost in port 8887)
```
python3 start_flask.py &
```
The & at the end of the is not mandatory, but it makes the script to run on the background. 

If you are using Raspberry PI, use this command instead:
```
python3 start_flask_RPi.py &
```

5. Connect to the server from a browser:
If you are running to 
```
http://your_ip_addres:8887
```


