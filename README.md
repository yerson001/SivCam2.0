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

6. Accessing your surveillance camera from outside your network. 

6.1 Setup a Dynamic DNS 
To access your internal LAN, the best option is to use Dynamic DNS or DDNS. This is basically a service to map the WAN IP address that you currently have, to a name of your choice.
There are several free options, for instance noip.com or dynu.com, offer this service for free. Just sign up, choose a name and domain and associate it with your current WAN IP address.  

You could access your LAN using your WAN IP address, that you can easily get visiting whatsmyip.org if you are visiting that page from home.
The problem is that from time to time this WAN IP changes, and if you are not at home you have no way to know it unless you tell someone at home to visit that page for you and text you the address.
To avoid this problem, DDNS service will associate a name, for example "myawesomehome.ddns.net" to your WAN IP. 

6.2 Activate a DynDNS service in your router/LAN
It's not enough with the first step. To keep the name linked to the right WAN IP, you have to notify the DNS server if there is a change of IP. 
To do this almost all routers offer the possibility to subscribe to a DynDNS service. 
So, look for DDNS or DynDNS in your router and add one service with the following information:
- Provider: noip.com or dynu.com or the provider of your choice 
- username: yourusername 
- password: yourpassword 
- Domain-name: myawesomehome.ddns.net 

Don't forget to activate the service and save that preferences. 

6.2 Activate port forwarding 
Finally you need to setup one more thing in your router. You have to tell your router that any incoming query 
with destination to port 8887 should go to your raspberry Pi (or the computer where you installed your 
surveillance). For that you need to look for "port forwarding" in your router and add one rule
with these info:
- Computer: IP address of where your camera is (for example 192.168.1.15)
- Incoming port: 8887
- Outgoing port: 8887
- Protocol: TCP 
If there are more fields, leave them blank or with default values. 

For a more detailed explanation see this article. 
https://www.howtogeek.com/66438/how-to-easily-access-your-home-network-from-anywhere-with-ddns/

This way you can access your camera from anywhere using browser and this address: 
```
http://myawesomehome.ddns.net:8887
```


Notes: the main script usage is the following  

```
usage: start_flask.py [-h] [--port PORT] [--host HOST] [-p CAMERA_TYPE]
                      [--area AREA] [--delay DELAY]

optional arguments:
  -h, --help      show this help message and exit
  --port PORT     socket port
  --host HOST     destination host name or ip
  -p CAMERA_TYPE  use piCamera
  --area AREA     minimum detected area to save a frame
  --delay DELAY   minimum delay to save a frame again
  --AI            if given, uses tensorflow model to detect people
```
## Extra

There are two extra scripts to do just video streaming:

- video_stream_receive.py
- video_stream_send.py  

Those ones do not use Flask, but send video signal directly to another PC. Can be useful for some applications. 
Important to know is that video_stream_receive.py acts as server and should be started first to start listening to incoming connections. 

## License 
This is opened with a MIT License, so use it as you want.    

## Acknowledgements

Some code for Flask integration and to do motion detection are taken from the great PyImageSearch blog by Adrian Rosebrock. 


