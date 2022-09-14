# ###################################################################
# Ruben Cardenes -- Apr 2020
#
# File:        video_stream_receive.py
# Description: This script starts a server that listen from incoming video streaming
#              connection (for instance from a Raspberry Pi), shows the video and
#              then depending on the mode it does:
#
#               Note: This script starts the server that listen to incoming streaming connections
#                     and should be started before video_stream_send.py
#
# ###################################################################

import cv2
import os
import numpy as np
from threading import Thread
import socket
from argparse import ArgumentParser
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class VideoClientThread(Thread):
    """Class to Receive video data from client"""

    def __init__(self, ip, port, connection):
        Thread.__init__(self)
        self.ip = ip
        self.port = port
        self.connection = connection

        print("[+] New server socket thread started for " + ip + ":" + str(port))

    def run(self):
        stream_bytes = b' '
        cv2.namedWindow("video feed", cv2.WINDOW_KEEPRATIO)

        # stream video frames one by one
        try:
            print("Video thread started")
            frame_num = 0
            i = 0
            while True:
                i += 1
                stream_bytes += self.connection.recv(1024)
                first = stream_bytes.find(b'\xff\xd8')
                last = stream_bytes.find(b'\xff\xd9')
                if first != -1 and last != -1:
                    jpg = stream_bytes[first:last + 2]
                    stream_bytes = stream_bytes[last + 2:]
                    image = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                    frame_num += 1
                    cv2.imshow("video feed", image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            cv2.destroyAllWindows()

        finally:
            self.connection.close()
            print("Connection closed on thread 1")


def start_multihreaded_server(server_host, port):

    TCP_IP = server_host
    TCP_PORT = port

    tcpServer = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcpServer.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    tcpServer.bind((TCP_IP, TCP_PORT))
    threads = []

    # Video connection
    tcpServer.listen(4)
    print("Python server : Waiting for Video connection from TCP clients...")
    (conn, (ip, port)) = tcpServer.accept()
    newthread = VideoClientThread(ip, port, conn)
    newthread.start()
    threads.append(newthread)

    for t in threads:
        t.join()


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--port', type=int,
                        dest='port',
                        default=8887,
                        help='socket port',
                        required=False)
    parser.add_argument('--host', type=str,
                        dest='host',
                        default='0.0.0.0',
                        help='destination host name or ip',
                        required=False)
    args = vars(parser.parse_args())

    server_host = args['host']
    port = args['port']

    start_multihreaded_server(server_host, port)

