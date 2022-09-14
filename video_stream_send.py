# ###################################################################
# Ruben Cardenes -- Apr 2020
#
# File:        video_stream_send.py
# Description: This scripts starts a client in the Raspberry pi that connects
#              to a server in another PC. Upon connection, this scripts sends
#              a video stream from the PI camera encoded as JPEG
#
#               Note: This script starts the client and should be started
#                     after video_stream_receive.py
# ###################################################################

import io
import socket
import struct
import time
import numpy as np
import os
from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import pickle
from threading import Thread
from argparse import ArgumentParser
from MultiObjectMotionDetection import MultiObjectMotionDetector

# Video Sending Thread
class VideoSendThread(Thread):
    # A class to send video frames using threads
    # This class inherits from Thread, which means that will run on a separate Thread
    # whenever called, it starts the run method

    def __init__(self, host, port, camera_resolution=(640, 480)):
        Thread.__init__(self)
        # create socket and bind host
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((host, port))
        self.connection = self.client_socket.makefile('wb')
        self.camera_resolution = camera_resolution
        self.stopped = False
        self.md = MultiObjectMotionDetector()
        self.output_name = ""
        self.save_time = 0

    def save_frame(self, frame):
        date_time = time.strftime("%m_%d_%Y-%H:%M:%S")
        self.output_name = os.path.join("./data/images", date_time + ".jpg")
        self.save_time = time.time()
        cv2.imwrite(self.output_name, frame)
        cv2.putText(frame, "Saving", (0, 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1)

    def run(self):
        try:

            with PiCamera() as camera:
                camera.resolution = self.camera_resolution  # pi camera resolution
                camera.framerate = 10  # 10 frames/sec
                time.sleep(2)  # give 2 secs for camera to initialize
                rawCapture = PiRGBArray(camera, size=self.camera_resolution)
                stream = camera.capture_continuous(rawCapture,
                                                    format="bgr", use_video_port=True)

                # send jpeg format video stream
                for f in stream:
                    frame = f.array
                    rawCapture.truncate(0)

                    # MOTION DETECTION
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray_blur = cv2.GaussianBlur(gray, (7, 7), 0)
                    thresh = np.zeros(frame.shape)
                    self.md.update(gray_blur)
                    thresh, md_boxes = self.md.detect(gray_blur)
                    if md_boxes is not None:
                        total_area = 0
                        for b in md_boxes:
                            cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]),
                                          (0, 0, 255), 1)
                            total_area += (b[2] - b[0])*(b[3] - b[1])
                        if total_area > 500:
                            # print("boxes: ", md_boxes)
                            if time.time() - self.save_time > 3:
                                self.save_frame(frame)

                    # if the thread indicator variable is set, stop the thread
                    # and resource camera resources
                    if self.stopped:
                        stream.close()
                        rawCapture.close()
                        camera.close()
                        return

                    (flag, encodedImage) = cv2.imencode(".jpg", frame)

                    # ensure the frame was successfully encoded
                    if not flag:
                        continue

                    # Add header to frame in byte format
                    bytes_to_send = (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                       bytearray(encodedImage) + b'\r\n')

                    # Send frame
                    self.connection.flush()
                    self.connection.write(bytes_to_send)

            # Pack zero as little endian unsigned long and send it to signal end of connection
            self.connection.write(struct.pack('<L', 0))

        finally:
            self.connection.close()
            self.client_socket.close()

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--port', type=int,
                        dest='port',
                        default=8887,
                        help='socket port',
                        required=False)
    parser.add_argument('--host', type=str,
                        dest='host',
                        default='192.168.1.3',
                        help='destination host name or ip',
                        required=False)
    args = vars(parser.parse_args())

    host = args['host']
    port = args['port']

    threads = []

    newthread = VideoSendThread(host, port)
    newthread.start()
    threads.append(newthread)

    for t in threads:
        t.join()
