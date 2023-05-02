#!/usr/bin/env python3
"""
 Copyright (C) 2018-2021 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import colorsys
import logging
import os
import random
import sys
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path
from time import perf_counter
from PIL import Image, ImageFont, ImageDraw
import cv2
import numpy as np
from openvino.inference_engine import IECore
from configparser import ConfigParser
from pathlib import Path
from paho.mqtt import client as mqtt_client
import random
import time
from datetime import datetime
import os
import math
import json
from ping3 import ping, verbose_ping
import gc

from multiprocessing import Process, Queue, Pool, Pipe
import sys
from threading import Timer
import time
import imutils

from MyClass import ParkingSpace
# from __future__ import print_function
from sort import *
# from scipy.optimize import linear_sum_assignment as linear_assignment
# from numba import jit
import numpy as np
# from sklearn.utils.l \
# import linear_assignment
from filterpy.kalman import KalmanFilter

# python3 object_detection_cht.py -at ssd -i rtsp://admin:admin@10.100.101.1:554/ch01/0 -m xml/pedestrian-and-vehicle-detector-adas-0001.xml -d MYRIAD
# 指定import到 ../common/python
# sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python'))
sys.path.append('./common/python')
import models
import monitors
from pipelines import get_user_config, AsyncPipeline
from images_capture import open_images_capture
from performance_metrics import PerformanceMetrics
from helpers import resolution

from dotenv import load_dotenv
from myPing import ping_some_ip

logging.basicConfig(format='[ %(levelname)s ] %(message)s', level=logging.INFO, stream=sys.stdout)
log = logging.getLogger()

'''new input '''
# c_IsDrawing = False  # true if mouse is pressed
# c_myPoint1 = (500, 300)
# c_myPoint2 = (600, 400)
c_line_index = 0  # 目前是那一條線

# 四條線
# c_line0_s = (399, 321)
# c_line0_e = (76, 437)
# c_line1_s = (417, 334)
# c_line1_e = (379, 504)
# c_line2_s = (866, 318)
# c_line2_e = (1091, 459)
# c_line3_s = (888, 305)
# c_line3_e = (1273, 357)

# C_TOTAL_EVENT_COUNT = 8
# 綠色框框的座標點
# c_area_green = [(0, 150), (300, 150), (300, 250), (0, 250)]

# 黃色框框的座標點
# c_area_yellow = [(350, 250), (600, 250), (600, 350), (350, 350)]
# c_area_left = [(500, 200), (600, 350), (300, 250), (300, 250)]

# broker = '10.190.0.67'  # mqtt 的ip
# broker = '127.0.0.1'
port = 1883  # mqtt port

# load_dotenv() #讀取設定檔中的內容至環境變數
load_dotenv('cht_people.env')
print(os.getenv('positionId'))

positionid = int(os.getenv('positionId'))
inAngle = int(os.getenv('inAngle'))  # 進的角度

myconfig = {'cameraId': ["LU-CC08", "LU-CC09", "LU-CC17", "DB-CC04",
                         "LY-CC03", "LY-CC04", "LY-CC07", "LY-CC08",
                         'tcar'
                         ],
            'rtspUrl': [
                'rtsp://10.190.1.36/stream1',
                'rtsp://syno:37c0f6822733fbfb8faf441ebea9b039@10.190.0.66:554/Sms=43.unicast',
                'rtsp://10.190.1.68/stream1',
                'rtsp://10.190.3.18/stream1',
                'rtsp://10.190.3.145/stream1',
                'rtsp://10.190.3.146/stream1',
                'rtsp://syno:0429bd6b5a03157da4a92b13313c3055@10.190.0.66:554/Sms=34.unicast',
                'rtsp://syno:4c423e0b4a8a08b1ee05b020eae18adf@10.190.0.66:554/Sms=35.unicast',
                'rtsp://admin:admin@10.100.101.1:554/ch00/0',
                # '2022-08-19_11-07-56.mp4',
            ],
            'cameraHost': ['10.190.1.36', '10.190.0.66', '10.190.1.68', '10.190.3.18',
                           '10.190.3.145', '10.190.3.146', '10.190.0.66', '10.190.0.66',
                           '10.100.101.1'
                           ],
            'mqttServer': [
                '10.190.0.67', '10.190.0.67', '10.190.0.67', '10.190.0.67',
                '10.190.0.67', '10.190.0.67', '10.190.0.67', '10.190.0.67',
                '127.0.0.1',
            ],
            }
cameraId = myconfig['cameraId'][positionid]
rtspUrl = myconfig['rtspUrl'][positionid]
cameraHost = myconfig['cameraHost'][positionid]
broker = myconfig['mqttServer'][positionid]
print(f'{cameraId} __ {cameraHost} __ {rtspUrl} __mqtt: {broker}')


filename = Path("./xml/person-vehicle-bike-detection-crossroad-0078.xml")
# filename = Path("sunskyN2583D-SK_2021_05_07_13_51_49.mp4")
# filename = Path("/home/kumiusge/source/tcar/trafficLight/sunskyN2583D-SK_2021_05_07_13_51_49.mp4")
# 測試攝影機的rtsp路徑
# rtspUrl = "rtsp://10.100.101.121/onvif-media/media.amp?streamprofile=Profile1&audio=0"
# rtspUrl = "rtsp://admin:admin@10.100.101.1:554/ch00/0"
# rtspUrl = ("/home/kumiusge/source/EastRiftValley/openvino/test/python/video/parkingSpace1.mp4")
# rtspUrl = ("/home/kumiusge/source/EastRiftValley/openvino/test/python/video/people.mp4")
# filename = Path("sunskyN2583D-SK_2021_05_07_13_51_49.mp4")

# rtspUrl = '2022-08-19_11-07-56.mp4'
mqtt_run_cycle_sec = 30
next_send_mqtt_time = 0
# mqtt 的發佈topic
# mytopic = "parkingSpace/"
mytopic = "pedestrianCount/"

lastCarPosition = -1  # 車輛的上次位置，-1：不在區域裏；0：綠燈線；1：黃燈線。
# log的儲存路徑
S_DATA_FILE_PATH = "./log"

# log的檔名
S_LOG_FILE = None

# 設定檔的位置（記錄兩個框框的座標）
S_SETTING_FILE = "./mycartrack.ini"
cfg = ConfigParser()

notriggerCount = 0  # 沒偵測到的次數

# generate client ID with pub prefix randomly
client_id = f'python-mqtt-{random.randint(0, 1000)}'
myMqttClient = None
print(cv2.__version__)  # 印出opencv 的版本
# filename = Path("~/source/tcar/trafficLight/cars.mp4")

if not filename.exists():
    print("Oops, file doesn't exist!")
else:
    print("Yay, the file exists!")
    print()

# 設定影像尺寸
width = 1024
height = 768

parkingSpaces = {}  # 停車格
posId = ["001"]  # 只要用一個大圈，每週期記算裏面人數，離開人數。
# posId = ["001", "002", "003"]
C_TOTAL_EVENT_COUNT = len(posId) * 4

# process_parent_conn= None   #process com
# process_child_conn = None

tracker = Sort(max_age=4, min_hits=4)
boxes = []
indexIDs = []
memory = {}
previous = {}
previous3 = {}
previous4 = {}
previous5 = {}
boxAngles = {}
boxInParkingSpacesPosition = {}

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")

inside_count = 0  # 內圈人數
exit_count = 0  # 離開人數

isPeopleActivate = 0  # 是否有人在動


recordFileProcess_Count=200  #配合recordFileProcess（）
recordFileProcess_CountMax=200
# def ping_some_ip(host, src_addr=None):
#     second = ping(host, src_addr=src_addr,timeout=2)
#     if (second is None):  # ping兩次比較好。
#         second = ping(host, src_addr=src_addr)
#     return second


def isOverLap(self, indexIDs, boxes, x1, y1, x2, y2, id):
    jdg = False
    # if(id in self.overlapNotUsedId):  # 不能這樣做，因為不要的可能回覆。高速公路主線要用
    #     msg = "number={},is not used id.".format(id)
    #     setSystemMessage(msg)
    #     return True
    imageArea1 = (x2 - x1) * (y2 - y1)  # 影像面積
    center1 = (int(x1 + (x2 - x1) / 2), int(y1 + (y2 - y1) / 2))
    areaDiffPercentage = 0  # 面積相差百分比
    twoPointDistance = 0  # 兩點位直線距離

    index = 0
    for box in boxes:
        # extract the bounding box coordinates
        (x, y) = (int(box[0]), int(box[1]))
        (w, h) = (int(box[2]), int(box[3]))  # 其實w是第二個點的x. h是第二點的y
        imageArea = (w - x) * (h - y)  # 影像面積
        center = (int(x + (w - x) / 2), int(y + (h - y) / 2))
        if (imageArea1 > imageArea):
            areaDiffPercentage = (
                                         imageArea1 - imageArea) / imageArea1 * 100
        else:
            areaDiffPercentage = (imageArea - imageArea1) / imageArea * 100
        twoPointDistance = self.get_length(center1, center)
        twoPointDistancePercentage = self.get_length(
            center1, center) / (x2 - x1) * 100

        if (areaDiffPercentage < 20 and twoPointDistance < 30):
            previousId = int(box[4])
            if (previousId > id):
                del boxes[index]
                indexIDs.remove(previousId)
                if (previousId not in self.overlapNotUsedId):
                    self.overlapNotUsedId.append(previousId)
                jdg = False
                msg = "find car overlap the car1 {},{},{},{} , car2 {},{},{},{}, the number={}, {},{} del id:{}, save id:{}".format(
                    x1, y1, x2, y2, x, y, w, h, areaDiffPercentage, twoPointDistance, twoPointDistancePercentage,
                    previousId, id)
            else:
                jdg = True
                msg = "find car -jdg = True- overlap the car1 {},{},{},{} , car2 {},{},{},{}, the number={}, {},{} del id:{}, save id:{}".format(
                    x1, y1, x2, y2, x, y, w, h, areaDiffPercentage, twoPointDistance, twoPointDistancePercentage, id,
                    previousId)
            setSystemMessage(msg)
            break
        index = index + 1
    return jdg


# 連線mqtt
def connect_mqtt():
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(client_id)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client


# 在圖片中劃出兩個框框的位置
def draw_triggerArea(event, x, y, flags, image):
    # global c_IsDrawing  # true if mouse is pressed
    global c_line_index
    # global c_area_green
    # global c_area_yellow

    # isLast = False
    #
    # cv2.rectangle(image,(0,0),(100,100),(0,255,255),2)
    pos = (x, y)
    i = c_line_index % C_TOTAL_EVENT_COUNT  # 20200421 add
    # i = c_line_index % 4
    if event == cv2.EVENT_LBUTTONDOWN:
        # c_IsDrawing = True
        j = i % 4
        index = int(i / 4)
        parkingSpaces[posId[index]].c_area[j] = pos

        # if (i == 0):
        #     c_area_green[0] = pos
        # elif (i == 1):
        #     c_area_green[1] = pos
        # elif (i == 2):
        #     c_area_green[2] = pos
        # elif (i == 3):
        #     c_area_green[3] = pos
        #
        # elif (i == 4):
        #     c_area_yellow[0] = pos
        # elif (i == 5):
        #     c_area_yellow[1] = pos
        # elif (i == 6):
        #     c_area_yellow[2] = pos
        # elif (i == 7):
        #     c_area_yellow[3] = pos
        # c_IsDrawing = False
        writeSettingFile(S_SETTING_FILE)

    elif event == cv2.EVENT_LBUTTONUP:
        # if i == 0:
        #     c_line0_e = pos
        # elif i == 1:
        #     c_line1_e = pos
        c_line_index += 1
        # c_IsDrawing = False
        # 寫入 This is a testing! 到檔案
        # print >>fp, "This is a testing!"
        msg = ""
        # if i == C_TOTAL_EVENT_COUNT - 1:
        #     isLast = True
        # return isLast


# 本次不用
def drawMyLine(image, index=-1):
    global c_line1_s, c_line1_e
    global c_line2_s, c_line2_e
    global c_line3_s, c_line3_e
    global c_line0_s, c_line0_e
    pt1 = (0, 0)
    pt2 = (0, 0)
    mycolor = (0, 0, 0)
    i = 0
    if index == -1:
        for i in range(2):
            if i == 0:
                pt1 = c_line0_s
                pt2 = c_line0_e
                mycolor = (255, 0, 0)
            elif i == 1:
                pt1 = c_line1_s
                pt2 = c_line1_e
                mycolor = (255, 255, 0)
            cv2.line(image, pt1, pt2, mycolor, 5)
    else:
        i = index
        if (i == 0):
            pt1 = c_line0_s
            pt2 = c_line0_e
            mycolor = (255, 0, 0)
        elif (i == 1):
            pt1 = c_line1_s
            pt2 = c_line1_e
            mycolor = (255, 255, 0)
        cv2.line(image, pt1, pt2, mycolor, 5)
    return image


# 在圖片中劃出兩個框框的位置
def drawMyArea(image, index=-1):
    i = 0
    if index == -1:
        for j in range(len(posId)):
            tmpcolor = parkingSpaces[posId[j]].color
            a = parkingSpaces[posId[j]].c_area
            cv2.line(image, a[0], a[1], tmpcolor, 3)
            cv2.line(image, a[1], a[2], tmpcolor, 3)
            cv2.line(image, a[2], a[3], tmpcolor, 3)
            cv2.line(image, a[3], a[0], tmpcolor, 3)
    else:
        tmpcolor = parkingSpaces[posId[index]].color
        a = parkingSpaces[posId[index]].c_area
        cv2.line(image, a[0], a[1], tmpcolor, 3)
        cv2.line(image, a[1], a[2], tmpcolor, 3)
        cv2.line(image, a[2], a[3], tmpcolor, 3)
        cv2.line(image, a[3], a[0], tmpcolor, 3)

    return image


# mqtt發佈訊息
def publish(client, topic, sendMsg):
    msg_count = 0
    # while True:
    # time.sleep(1)
    # sendMsg = f"messages: {msg_count}"
    result = client.publish(topic, sendMsg)
    # result: [0, 1]
    status = result[0]
    if status == 0:
        setSystemMessage(f"Send `{sendMsg}` to topic `{topic}`")
    else:
        setSystemMessage(f"Failed to send message to topic {topic}")
    msg_count += 1
    # time.sleep(1)


# value = dict(id='The Breakfast Club',status=1)
def sendMqtt(mqttclient, topic, parkingSpaces):
    now = datetime.now()
    busId = 'vid001'
    # title = "vehicle/report/{}".format(busId)
    title = mytopic
    print(title)
    angle = 200
    # 設置日期時間的格式
    ISOTIMEFORMAT = '%Y/%m/%d %H:%M:%S'

    #     while True:
    #         angle = random.randint(-500,500)
    #     t = datetime.now().strftime(ISOTIMEFORMAT)
    t = datetime.now().timestamp() * 1000
    #         t = int(t)
    #         print(t)
    #    payload = {'Temperature' : t0 , 'Time' : t}
    mymsg = "no name to match mqtt service. error."
    payload = None
    if (topic == 'parkingSpace/'):
        status = []
        for ps in posId:
            status.append({'sid': ps,
                           'status': parkingSpaces[ps].isOccStatus,
                           'timestamp': parkingSpaces[ps].timestamp})
        payload = {
            "cameraId": cameraId,
            "status": status

        }
    elif (topic == 'pedestrianCount/'):
        status = []
        ps = posId[0]
        status.append({'sid': 'entrance',
                       'count': parkingSpaces[ps].inDirCount,
                       'timestamp': parkingSpaces[ps].timestamp})
        status.append({'sid': 'exit',
                       'count': parkingSpaces[ps].outDirCount,
                       'timestamp': parkingSpaces[ps].timestamp})
        payload = {
            "cameraId": cameraId,
            "status": status
        }
    else:
        print(" The topic is Error.")
    if (parkingSpaces[ps].inDirCount > 0 or parkingSpaces[ps].outDirCount > 0):
        print(f'-------------{parkingSpaces[ps].inDirCount}, {parkingSpaces[ps].outDirCount}----------------')
    '''
    ic: parkingSpace/
    Data:
    {
    	cameraId:string,		//攝影機編號
    	status:[
    		{
    			"sid": int32,     	//車位編號
    			"status": int32,		//狀態：0-無佔用；1-佔用。
    			"timestamp":uint64,	//時間
    			"licensePlate":string,	//車號(擴充功能)
    		}
    	]
    }
    Topic: pedestrianCount/
    Data:
    {
    	cameraId:string,		//攝影機編號
    	status:[
    		{
    			"areaid": int32,	//畫面中區域編號
    			"count": int32,		//人數。
    			"timestamp":uint64,	//時間
    		}
    	]

    {
        cameraId:"SC01-C02",
        status:[
            {"sid": "entrance","count": 1,"timestamp":1646633602},
            {"sid": "exit","count": 3,"timestamp":1646633601}
        ]
    }    	
    '''

    if (payload is not None):
        print(json.dumps(payload))
        # 要發布的主題和內容
        mqttclient.publish(title, json.dumps(payload))
        setSystemMessage(title + '  ' + json.dumps(payload) + '\n')
        mymsg = 'send mqtt success. '
        setSystemMessage(mymsg + '\n')
        time.sleep(1)

    else:
        print(mymsg)
        setSystemMessage(mymsg + '\n')


# 計算x是否在abcd四點之中。一定要用順時針排列。最少要三個點。
def isInArea(x, A, B, C, D=None, E=None, F=None):
    jdg = False
    a = [0, 0, 0, 0, 0, 0]
    a[0] = pAngle(x, A, B)
    a[1] = pAngle(x, B, C)
    if D != None:
        a[2] = pAngle(x, C, D)
        if E != None:
            a[3] = pAngle(x, D, E)
            if F is not None:
                a[4] = pAngle(x, E, F)
                a[5] = pAngle(x, F, A)
            else:
                a[4] = pAngle(x, E, A)
        else:
            a[3] = pAngle(x, D, A)
    else:
        a[2] = pAngle(x, C, A)
    total = 0
    for i in range(6):
        total += a[i]
    if total > 358:
        jdg = True
        # print('------角度總和：{},點位：{}, 六角度：{}----------in the area:'.format(str(total), x, a))
    # else:
    # print('----------------out of the area:', str(total),x,a)
    return jdg


# p1:start point, p2:end point
def getAzimuth(p1, p2):
    north = (p1[0], 0)
    angle = pAngle(p1, north, p2)
    xy = 0  # 第幾象限，配合方位角，小到大，1-4。
    if p1[0] < p2[0] and p1[1] > p2[1]:
        xy = 1
    elif p1[0] < p2[0] and p1[1] < p2[1]:
        xy = 2
    elif p1[0] > p2[0] and p1[1] < p2[1]:
        xy = 3
    elif p1[0] > p2[0] and p1[1] > p2[1]:
        xy = 4

    if xy > 2:
        angle = 360 - angle
    return angle


# 用三個點，計算角度
def pAngle(cen, first, second):
    M_PI = 3.1415926535897
    ma_x = first[0] - cen[0]
    ma_y = first[1] - cen[1]
    mb_x = second[0] - cen[0]
    mb_y = second[1] - cen[1]
    v1 = (ma_x * mb_x) + (ma_y * mb_y)
    ma_val = math.sqrt(ma_x * ma_x + ma_y * ma_y)
    mb_val = math.sqrt(mb_x * mb_x + mb_y * mb_y)
    angleAMB = 0
    if (ma_val * mb_val) == 0:
        angleAMB = 0
    else:
        cosM = v1 / (ma_val * mb_val)
        try:
            angleAMB = math.acos(cosM) * 180 / M_PI
        except Exception as e:
            print('error.' + str(e))
            print(f' math.acos(cosM) failure. cosM = {cosM}')
            angleAMB = 0
    return angleAMB


# 顯示及記錄系統訊息
def setSystemMessage(msg):
    global S_DATA_FILE_PATH
    global S_LOG_FILE
    createFolder(S_DATA_FILE_PATH)
    # 開啟檔案
    if S_LOG_FILE == None:
        S_LOG_FILE = "{}/{}.txt".format(
            S_DATA_FILE_PATH, time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
    fp = open(S_LOG_FILE, "a")
    # 格式化成2016-03-20 11:45:39形式
    mydate = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    # 關閉檔案
    # print >> fp, msg
    msg1 = "{} -> {}\n".format(mydate, msg)
    fp.writelines(msg1)
    print(msg1)
    fp.close()


# 讀取ini裏的設定值
def readSettingFile(initFile, parkingSpaces):
    global cfg

    # global c_area_green
    # global c_area_yellow

    cfg.read(initFile)
    if 'default' in cfg:
        # tmp = cfg['default'].get('c_line0_s')
        # if(tmp is not None):
        #     c_line0_s = parseToTurper(tmp)  # ok
        # tmp = cfg['default'].get('c_line0_e')
        # if(tmp is not None):
        #     c_line0_e = parseToTurper(tmp)  # ok
        for i in range(len(posId)):
            ss = "p" + str(i)
            tmp = cfg['default'].get(ss)
            if tmp is not None:
                parkingSpaces[posId[i]].c_area = parseRectangleToTurper(tmp)

        # tmp = cfg['default'].get('c_motor_largest_imageArea')
        # if(tmp is not None):
        #     c_motor_largest_imageArea = parseAreaToArray(tmp)


# 寫入ini裏的設定值
def writeSettingFile(initFile):
    global cfg

    # global c_area_green
    # global c_area_yellow
    # try:
    if ('default' in cfg):
        for i in range(len(posId)):
            cfg['default']['p' + str(i)] = str(parkingSpaces[posId[i]].c_area)
        # cfg['default']['c_area_green'] = str(c_area_green)
        # cfg['default']['c_area_yellow'] = str(c_area_yellow)
    else:
        cfg.add_section('default')
        for i in range(len(posId)):
            cfg.set('default', 'p' + str(i), str(parkingSpaces[posId[i]].c_area))
        # cfg.set('default', 'c_area_green', str(c_area_green))
        # cfg.set('default', 'c_area_yellow', str(c_area_yellow))

        # cfg.set('default', 'c_motor_largest_imageArea',
        #         str(c_motor_largest_imageArea))
        # cfg.set('default', 'c_car_largest_imageArea',
        #         str(c_car_largest_imageArea))
    with open(initFile, 'w', encoding='cp950') as f:
        cfg.write(f)


# 把座標文字轉換成矩陣
def parseToTurper(tmp):
    tmp = tmp.replace("(", "")
    tmp = tmp.replace(")", "")
    words = tmp.split(",")
    return int(words[0]), int(words[1])


# 把四個點的座標文字轉換成矩陣
def parseRectangleToTurper(tmp):
    tmp = tmp.replace("[", "")
    tmp = tmp.replace("]", "")
    tmp = tmp.replace("(", "")
    tmp = tmp.replace(")", "")
    tmp = tmp.replace(" ", "")
    words = tmp.split(",")
    a = [(int(words[0]), int(words[1])), (int(words[2]), int(words[3])),
         (int(words[4]), int(words[5])), (int(words[6]), int(words[7]))]
    return a


# 新增一個目錄
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


class MyCar:
    id = ''  # class variable shared by all instances
    areaid = -1
    angle = -1
    classesid = -1
    imageArea = -1
    position = [0, 0, 0, 0]

    def __init__(self, id, areaid, angle, classesid, imagearea, position):
        self.id = id  # instance variable unique to each instance
        self.areaid = areaid
        self.angle = angle
        self.classesid = classesid
        self.imageArea = imagearea
        self.position = position


'''end input'''


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument('-m', '--model', help='Required. Path to an .xml file with a trained model.',
                      required=True, type=Path)
    args.add_argument('-at', '--architecture_type', help='Required. Specify model\' architecture type.',
                      type=str, required=True, choices=('ssd', 'yolo', 'yolov4', 'faceboxes', 'centernet', 'ctpn',
                                                        'retinaface', 'ultra_lightweight_face_detection',
                                                        'retinaface-pytorch'))
    args.add_argument('-i', '--input', required=True,
                      help='Required. An input to process. The input must be a single image, '
                           'a folder of images, video file or camera id.')
    args.add_argument('-d', '--device', default='CPU', type=str,
                      help='Optional. Specify the target device to infer on; CPU, GPU, HDDL or MYRIAD is '
                           'acceptable. The demo will look for a suitable plugin for device specified. '
                           'Default value is CPU.')

    common_model_args = parser.add_argument_group('Common model options')
    common_model_args.add_argument('--labels', help='Optional. Labels mapping file.', default=None, type=str)
    common_model_args.add_argument('-t', '--prob_threshold', default=0.5, type=float,
                                   help='Optional. Probability threshold for detections filtering.')
    common_model_args.add_argument('--keep_aspect_ratio', action='store_true', default=False,
                                   help='Optional. Keeps aspect ratio on resize.')
    common_model_args.add_argument('--input_size', default=(600, 600), type=int, nargs=2,
                                   help='Optional. The first image size used for CTPN model reshaping. '
                                        'Default: 600 600. Note that submitted images should have the same resolution, '
                                        'otherwise predictions might be incorrect.')
    common_model_args.add_argument('--anchors', default=None, type=float, nargs='+',
                                   help='Optional. A space separated list of anchors. '
                                        'By default used default anchors for model. Only for YOLOV4 architecture type.')
    common_model_args.add_argument('--masks', default=None, type=int, nargs='+',
                                   help='Optional. A space separated list of mask for anchors. '
                                        'By default used default masks for model. Only for YOLOV4 architecture type.')

    infer_args = parser.add_argument_group('Inference options')
    infer_args.add_argument('-nireq', '--num_infer_requests', help='Optional. Number of infer requests',
                            default=0, type=int)
    infer_args.add_argument('-nstreams', '--num_streams',
                            help='Optional. Number of streams to use for inference on the CPU or/and GPU in throughput '
                                 'mode (for HETERO and MULTI device cases use format '
                                 '<device1>:<nstreams1>,<device2>:<nstreams2> or just <nstreams>).',
                            default='', type=str)
    infer_args.add_argument('-nthreads', '--num_threads', default=None, type=int,
                            help='Optional. Number of threads to use for inference on CPU (including HETERO cases).')

    io_args = parser.add_argument_group('Input/output options')
    io_args.add_argument('--loop', default=False, action='store_true',
                         help='Optional. Enable reading the input in a loop.')
    io_args.add_argument('-o', '--output', required=False,
                         help='Optional. Name of the output file(s) to save.')
    io_args.add_argument('-limit', '--output_limit', default=1000, type=int,
                         help='Optional. Number of frames to store in output. '
                              'If 0 is set, all frames are stored.')
    io_args.add_argument('--no_show', default=False, help="Optional. Don't show output.", action='store_true')
    io_args.add_argument('--output_resolution', default=None, type=resolution,
                         help='Optional. Specify the maximum output window resolution '
                              'in (width x height) format. Example: 1280x720. '
                              'Input frame size used by default.')
    io_args.add_argument('-u', '--utilization_monitors', default='', type=str,
                         help='Optional. List of monitors to show initially.')

    input_transform_args = parser.add_argument_group('Input transform options')
    input_transform_args.add_argument('--reverse_input_channels', default=False, action='store_true',
                                      help='Optional. Switch the input channels order from '
                                           'BGR to RGB.')
    input_transform_args.add_argument('--mean_values', default=None, type=float, nargs=3,
                                      help='Optional. Normalize input by subtracting the mean '
                                           'values per channel. Example: 255 255 255')
    input_transform_args.add_argument('--scale_values', default=None, type=float, nargs=3,
                                      help='Optional. Divide input by scale values per channel. '
                                           'Division is applied after mean values subtraction. '
                                           'Example: 255 255 255')

    debug_args = parser.add_argument_group('Debug options')
    debug_args.add_argument('-r', '--raw_output_message', help='Optional. Output inference results raw values showing.',
                            default=False, action='store_true')
    return parser


class ColorPalette:
    def __init__(self, n, rng=None):
        assert n > 0

        if rng is None:
            rng = random.Random(0xACE)

        candidates_num = 100
        hsv_colors = [(1.0, 1.0, 1.0)]
        for _ in range(1, n):
            colors_candidates = [(rng.random(), rng.uniform(0.8, 1.0), rng.uniform(0.5, 1.0))
                                 for _ in range(candidates_num)]
            min_distances = [self.min_distance(hsv_colors, c) for c in colors_candidates]
            arg_max = np.argmax(min_distances)
            hsv_colors.append(colors_candidates[arg_max])

        self.palette = [self.hsv2rgb(*hsv) for hsv in hsv_colors]

    @staticmethod
    def dist(c1, c2):
        dh = min(abs(c1[0] - c2[0]), 1 - abs(c1[0] - c2[0])) * 2
        ds = abs(c1[1] - c2[1])
        dv = abs(c1[2] - c2[2])
        return dh * dh + ds * ds + dv * dv

    @classmethod
    def min_distance(cls, colors_set, color_candidate):
        distances = [cls.dist(o, color_candidate) for o in colors_set]
        return np.min(distances)

    @staticmethod
    def hsv2rgb(h, s, v):
        return tuple(round(c * 255) for c in colorsys.hsv_to_rgb(h, s, v))

    def __getitem__(self, n):
        return self.palette[n % len(self.palette)]

    def __len__(self):
        return len(self.palette)


def get_model(ie, args):
    input_transform = models.InputTransform(args.reverse_input_channels, args.mean_values, args.scale_values)
    common_args = (ie, args.model, input_transform)
    if args.architecture_type in ('ctpn', 'yolo', 'yolov4', 'retinaface',
                                  'retinaface-pytorch') and not input_transform.is_trivial:
        raise ValueError("{} model doesn't support input transforms.".format(args.architecture_type))

    if args.architecture_type == 'ssd':
        return models.SSD(*common_args, labels=args.labels, keep_aspect_ratio_resize=args.keep_aspect_ratio)
    elif args.architecture_type == 'ctpn':
        return models.CTPN(ie, args.model, input_size=args.input_size, threshold=args.prob_threshold)
    elif args.architecture_type == 'yolo':
        return models.YOLO(ie, args.model, labels=args.labels,
                           threshold=args.prob_threshold, keep_aspect_ratio=args.keep_aspect_ratio)
    elif args.architecture_type == 'yolov4':
        return models.YoloV4(ie, args.model, labels=args.labels,
                             threshold=args.prob_threshold, keep_aspect_ratio=args.keep_aspect_ratio,
                             anchors=args.anchors, masks=args.masks)
    elif args.architecture_type == 'faceboxes':
        return models.FaceBoxes(*common_args, threshold=args.prob_threshold)
    elif args.architecture_type == 'centernet':
        return models.CenterNet(*common_args, labels=args.labels, threshold=args.prob_threshold)
    elif args.architecture_type == 'retinaface':
        return models.RetinaFace(ie, args.model, threshold=args.prob_threshold)
    elif args.architecture_type == 'ultra_lightweight_face_detection':
        return models.UltraLightweightFaceDetection(*common_args, threshold=args.prob_threshold)
    elif args.architecture_type == 'retinaface-pytorch':
        return models.RetinaFacePyTorch(ie, args.model, threshold=args.prob_threshold)
    else:
        raise RuntimeError('No model type or invalid model type (-at) provided: {}'.format(args.architecture_type))


def draw_detections(frame, detections, palette, labels, threshold, output_transform):
    size = frame.shape[:2]
    global parkingSpaces
    global tracker
    global memory
    global previous
    global previous3
    global previous4
    global previous5
    global COLORS
    global boxAngles, boxInParkingSpacesPosition, isPeopleActivate

    frame = output_transform.resize(frame)
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    # global parkingSpaces
    t = datetime.now().timestamp()
    for pid in posId:
        # parkingSpaces[pid].isOcc = -1
        parkingSpaces[pid].timestamp = t
        # parkingSpaces[pid].insideCount=0
    dets = []
    for detection in detections:
        if detection.score > threshold:
            class_id = int(detection.id)
            color = palette[class_id]
            det_label = labels[class_id] if labels and len(labels) >= class_id else '#{}'.format(class_id)
            xmin = max(int(detection.xmin), 0)
            ymin = max(int(detection.ymin), 0)
            xmax = min(int(detection.xmax), size[1])
            ymax = min(int(detection.ymax), size[0])
            xmin, ymin, xmax, ymax = output_transform.scale([xmin, ymin, xmax, ymax])

            top = max(0, np.floor(ymin + 0.5).astype('int32'))
            left = max(0, np.floor(xmin + 0.5).astype('int32'))
            bottom = min(size[1], np.floor(ymax + 0.5).astype('int32'))
            right = min(size[0], np.floor(xmax + 0.5).astype('int32'))

            dets.append([left, top, right, bottom, detection.score])
            # dets.append([top,left,  bottom,right, detection.score])

    np.set_printoptions(
        formatter={'float': lambda x: "{0:0.3f}".format(x)})
    dets = np.asarray(dets)
    tracks = None
    tracks = tracker.update(dets)  # 記錄軌跡
    boxes = []
    indexIDs = []
    c = []
    previous5 = previous4.copy()  # 前面幾個點
    previous4 = previous3.copy()  # 前面幾個點
    previous3 = previous.copy()  # 前面幾個點
    previous = memory.copy()
    memory = {}
    # font = ImageFont.truetype(font='font/NimbusRoman-Italic.otf',
    #                           size=np.floor(3e-2 * image.size[1] +
    #                                         0.5).astype('int32'))
    # font = ImageFont.truetype(font='font/NimbusRoman-Italic.otf',
    #                           size=30)
    # raspberrypi use this
    font = ImageFont.truetype(font='/usr/share/fonts/truetype/dejavu/DejaVuSans-BoldOblique.ttf',
                              size=30)
    for track in tracks:
        boxes.append(
            [track[0], track[1], track[2], track[3], track[4]])
        indexIDs.append(int(track[4]))
        memory[indexIDs[-1]] = boxes[-1]
    msg = (
        '--Found {} can used boxes from tracks {}'.format(len(boxes), len(tracks)))
    if len(boxes) == 0:
        isPeopleActivate = 0
    else:
        isPeopleActivate = 1
        i = int(0)
        for box in boxes:
            # extract the bounding box coordinates
            (x, y) = (int(box[0]), int(box[1]))
            (w, h) = (int(box[2]), int(box[3]))  # 其實w是第二個點的x. h是第二點的y
            boxid = box[4]
            imageArea = (w - x) * (h - y)  # 影像面積

            # draw a bounding box rectangle and label on the image
            # color = [int(c) for c in COLORS[classIDs[i]]]
            # cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            mycolor = [int(c) for c in COLORS[int(boxid) % len(COLORS)]]
            color = (mycolor[0], mycolor[1], mycolor[2])
            # cv2.rectangle(image, (2, 3), (111,111), [255,2,2], 1)
            # cv2.rectangle(image, (2, 3), (111,111), color, 1)
            # cv2.rectangle(image, (x, y), (w, h), color, 2)
            draw.rectangle([(x, y), (w, h)], outline=color)
            draw.text((w, y - 5), str(int(boxid)),
                      fill=color, font=font, outline=color)
            p0 = (int(x + (w - x) / 2), int(y + (h - y) / 2))
            p1 = None
            p4 = None
            p6 = None
            p8 = None

            angle = 0
            length = 0
            # if(indexIDs == 440):
            #     length = 0
            if boxid in previous:
                previous_box = previous[boxid]
                (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                p1 = (int(x2 + (w2 - x2) / 2), int(y2 + (h2 - y2) / 2))
                # p0 = (int(x + (w - x) / 2), int(h ))  #移到下面，在路上
                # p1 = (int(x2 + (w2 - x2) / 2), int(h2 )) #移到下面，在路上
                # cv2.line(image, p0, p1, color,5)
                draw.line([p0, p1], fill=color, width=3)
                angle = getAzimuth(p1, p0)

                carPosition = -1  # 是否有在那個區域裏，0：right; 1:center; 2:left

                # 畫出之前的點位
                if boxid in previous3:
                    previous_box2 = previous3[boxid]
                    (x3, y3) = (int(previous_box2[0]),
                                int(previous_box2[1]))
                    (w3, h3) = (int(previous_box2[2]),
                                int(previous_box2[3]))
                    # p3 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/2))
                    p4 = (int(x3 + (w3 - x3) / 2), int(y3 + (h3 - y3) / 2))
                    # cv2.line(image, p4, p1, color, 3)
                    draw.line([p4, p1], fill=color, width=3)
                    angle = getAzimuth(p4, p0)
                    # length = get_length(p4, p0)
                    # # print("  --  draw previous3 ")
                if boxid in previous4:
                    previous_box3 = previous4[boxid]
                    (x4, y4) = (int(previous_box3[0]),
                                int(previous_box3[1]))
                    (w4, h4) = (int(previous_box3[2]),
                                int(previous_box3[3]))

                    # p5 = (int(x3 + (w3-x3)/2), int(y3 + (h3-y3)/2))
                    p6 = (int(x4 + (w4 - x4) / 2), int(y4 + (h4 - y4) / 2))
                    # cv2.line(image, p6, p4, color, 2)
                    if (p4 is not None):
                        draw.line([p6, p4], fill=color, width=3)
                    else:
                        draw.line([p6, p1], fill=color, width=3)
                    angle = getAzimuth(p6, p0)

                if boxid in previous5:
                    previous_box4 = previous5[boxid]
                    (x5, y5) = (int(previous_box4[0]),
                                int(previous_box4[1]))
                    (w5, h5) = (int(previous_box4[2]),
                                int(previous_box4[3]))
                    # p7 = (int(x4 + (w4-x4)/2), int(y4 + (h4-y4)/2))
                    p8 = (int(x5 + (w5 - x5) / 2), int(y5 + (h5 - y5) / 2))
                    # cv2.line(image, p8, p6, color, 1)
                    # cv2.line(image, p8, p6, color, 1)
                    if (p6 is not None):
                        draw.line([p8, p6], fill=color, width=3)
                    elif (p4 is not None):
                        draw.line([p8, p4], fill=color, width=3)
                    else:
                        draw.line([p8, p1], fill=color, width=3)
                    angle = getAzimuth(p8, p0)
            if boxid in boxInParkingSpacesPosition:
                # print(f'{boxid} find previous in parkingspace position: {boxInParkingSpacesPosition[boxid]}')
                angle = getAzimuth(boxInParkingSpacesPosition[boxid], p0)
                # print(f'{boxid} get the new angle: {angle}')
            boxAngles[boxid] = angle
            # print(f'  {boxid}  --angle:{angle}')

            for pid in posId:
                color = (0, 0, 255)  # parkingSpaces[posId[i]].color
                a = parkingSpaces[pid].c_area

                if isInArea(p0, a[0], a[1], a[2], a[3]):

                    parkingSpaces[pid].indexIds_now.append(boxid)
                    if (boxid in parkingSpaces[pid].indexIds):
                        # print(f'{posId[i]} area, the {boxid} is repeat.')
                        continue
                    boxInParkingSpacesPosition[boxid] = p0
                    carPosition = 0
                    # 寫入文字
                    # cv2.putText(frame, 'occupy', a[1], cv2.FONT_HERSHEY_SIMPLEX, 1, color, 4)
                    draw.line([a[0], a[1]], fill=color, width=2)
                    draw.line([a[1], a[2]], fill=color, width=2)
                    draw.line([a[2], a[3]], fill=color, width=2)
                    draw.line([a[3], a[0]], fill=color, width=2)
                    if boxid not in parkingSpaces[pid].indexIds:
                        parkingSpaces[pid].indexIds.append(boxid)
                    parkingSpaces[pid].insideCount += 1
                    # print(f'{posId[i]} area is occ, now is {parkingSpaces[pid].insideCount } person.')
                    # parkingSpaces[pid].isOccTimes += 1
                    # setSystemMessage("區壓佔。" + posId[i])
                    # else:
                    #     if(p1 !=None):
                    #         if isInArea(p1, a[0], a[1], a[2], a[3]):
                    #             parkingSpaces[pid].exitCount +=1

                    # print(f'{pid} is occ. ')
                    # 要發佈的訊息
                    sendMsg = mytopic
                    # mqtt發佈出去
                    # publish(mqttClient, mytopic, sendMsg)
                    lastCarPosition = 0
    for pid in posId:
        # parkingSpaces[pid].isOcc += parkingSpaces[pid].isOccTimes
        parkingSpaces[pid].insideCount_last = parkingSpaces[pid].insideCount
        color = (220, 220, 255)  # parkingSpaces[posId[i]].color
        if (parkingSpaces[pid].insideCount > 0):
            color = (0, 0, 255)  # parkingSpaces[posId[i]].color

        count = 0
        dirInCount = 0
        dirOutCount = 0
        # parkingSpaces[pid].inDirCount=0
        # parkingSpaces[pid].outDirCount=0

        for boxid in parkingSpaces[pid].indexIds:
            if boxid not in parkingSpaces[pid].indexIds_now:  # 離開框框
                count += 1
                if boxid not in parkingSpaces[pid].indexIds_exit:
                    parkingSpaces[pid].indexIds_exit.append(boxid)
                    isDirIn = False
                    dif = abs(boxAngles[boxid] - inAngle)
                    if (dif < 90):
                        isDirIn = True
                    elif 360 - dif < 90:
                        isDirIn = True
                    if isDirIn is True:
                        parkingSpaces[pid].inDirCount += 1
                        print(f'inDirCount :{boxid} - {parkingSpaces[pid].inDirCount}')

                    else:
                        parkingSpaces[pid].outDirCount += 1
                        print(f'outDirCount : {boxid} -  {parkingSpaces[pid].outDirCount}')

        parkingSpaces[pid].exitCount = count
        parkingSpaces[pid].indexIds_now = []

        a = parkingSpaces[pid].c_area
        # draw.text(a[1], f'{parkingSpaces[pid].insideCount}_{parkingSpaces[pid].exitCount}',
        #           fill=color, font=font, outline=color)
        draw.text(a[2], 'in_out',
                  fill=color, font=font, outline=color)
        draw.text(a[0], f'{parkingSpaces[pid].inDirCount}_{parkingSpaces[pid].outDirCount}',
                  fill=color, font=font, outline=color)
        # print(f'{parkingSpaces[pid].insideCount} _  {parkingSpaces[pid].exitCount}')
    result = np.asarray(image)  # 轉回opencv

    return result,len(boxes)


def print_raw_results(size, detections, labels, threshold):
    setSystemMessage(' Class ID | Confidence | XMIN | YMIN | XMAX | YMAX ')

    for detection in detections:
        if detection.score > threshold:
            xmin = max(int(detection.xmin), 0)
            ymin = max(int(detection.ymin), 0)
            xmax = min(int(detection.xmax), size[1])
            ymax = min(int(detection.ymax), size[0])
            class_id = int(detection.id)
            det_label = labels[class_id] if labels and len(labels) >= class_id else '#{}'.format(class_id)
            setSystemMessage('{:^9} | {:10f} | {:4} | {:4} | {:4} | {:4} '
                             .format(det_label, detection.score, xmin, ymin, xmax, ymax))


def job(parkingSpaces):
    print("job run")
    # value = {'id': parkingSpaces, 'status': 5}
    # parkingSpaces[posId[index]].c_area[j]
    global myMqttClient
    # global parkingSpaces

    print('cycle run...')
    if myMqttClient is not None:
        sendMqtt(myMqttClient, mytopic, parkingSpaces)
    else:
        print(f'mqtt client is None. prepare to connect the server:{broker}')
        myMqttClient = connect_mqtt()  # 連mqtt
        if myMqttClient is not None:
            # myMqttClient.loop_start()  # 連上後等待接收
            sendMqtt(myMqttClient, mytopic, parkingSpaces)

    # time.sleep(mqtt_run_cycle_sec)


def myInit():
    global parkingSpaces
    for ps in posId:
        t = datetime.now().timestamp() * 1000
        isOcc = 0  # 每週期更新一次
        c_area = [(0, 150), (300, 150), (300, 250), (0, 250)]
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        parkingSpaces[ps] = ParkingSpace(t, isOcc, c_area, color)
    readSettingFile(S_SETTING_FILE, parkingSpaces)
    setSystemMessage("讀設定檔：" + S_SETTING_FILE)
    # for pk in parkingSpaces:
    #     print(str(pk.c_area))


def main():
    global parkingSpaces
    myInit()

    # process_parent_conn, process_child_conn = Pipe()
    # processB = Process(target=job, args=(process_child_conn,))
    # processB.start()

    # processB = Process(target=job, args=())
    # processB.start()

    args = build_argparser().parse_args()
    if args.architecture_type != 'yolov4' and args.anchors:
        setSystemMessage('The "--anchors" options works only for "-at==yolov4". Option will be omitted')
    if args.architecture_type != 'yolov4' and args.masks:
        setSystemMessage('The "--masks" options works only for "-at==yolov4". Option will be omitted')

    setSystemMessage('Initializing Inference Engine...')
    ie = IECore()

    plugin_config = get_user_config(args.device, args.num_streams, args.num_threads)

    setSystemMessage('Loading network...')

    model = get_model(ie, args)

    detector_pipeline = AsyncPipeline(ie, model, plugin_config,
                                      device=args.device, max_num_requests=args.num_infer_requests)

    cap = None
    video_writer = None
    # cap = open_images_capture(rtspUrl.input, args.loop)
    # cap = open_images_capture(rtspUrl, args.loop)
    curr_dt = datetime.now()  # 2021-08-25 15:04:33.794484
    nowStr = curr_dt.strftime('%Y-%m-%d_%H-%M-%S')
    recordFile = f'record_{nowStr}.avi'
    try:

        cap = cv2.VideoCapture(rtspUrl)
        # cap = cv2.VideoCapture(args.input)
        video_writer = cv2.VideoWriter()
    except Exception as ex:
        print(ex)

    next_frame_id = 0
    next_frame_id_to_show = 0

    setSystemMessage('Starting inference...')
    print("To close the application, press 'CTRL+C' here or switch to the output window and press ESC key")

    palette = ColorPalette(len(model.labels) if model.labels else 100)
    metrics = PerformanceMetrics()
    presenter = None
    output_transform = None

    '''新增的'''
    global lastCarPosition
    global notriggerCount
    # global myMqttClient
    # myMqttClient = connect_mqtt()  # 連mqtt
    # myMqttClient.loop_start()  # 連上後等待接收
    cv2.namedWindow('Detection Results', cv2.WINDOW_AUTOSIZE)
    imageCount = 0
    # t = Timer(10, job)  # 在5秒後，自動執行 hello()
    # t.start()

    next_send_mqtt_time = int(time.time()) + mqtt_run_cycle_sec
    processB = None


    while True:
        # frame = cap.read()
        # if frame is None:
        #     if next_frame_id == 0:
        #         raise ValueError("Can't read an image from the input")
        #     continue
        # else:
        #     cv2.imshow('image_display',frame)
        #     cv2.waitKey(100)

        # print('ping @ {}'.format(datetime.now()))
        if (imageCount % 100 == 0):
            result = ping_some_ip(cameraHost)
            # print(f'ping ->{result}')
            if result is None:
                print(f'ping {cameraHost} ping 失败！')
                if cap is not None:
                    cv2.destroyWindow('Detection Results')
                    cap.release()
                    cap = None
                time.sleep(30)
                continue
            else:
                # print('ping-{}成功，耗时{}s'.format(cameraHost, result))
                if cap is None:
                    try:
                        cap = cv2.VideoCapture(rtspUrl)
                        cv2.namedWindow('Detection Results', cv2.WINDOW_AUTOSIZE)
                    except Exception as e:
                        print('error.' + str(e))
                        print(f' connect the camera failure. {rtspUrl}')
                        time.sleep(30)
                        continue
        '''new '''
        isCarShow = False
        imageCount = imageCount + 1
        try:
            ret, frame = cap.read()
            if ret is False or frame is None:
                # if next_frame_id == 0:
                #     raise ValueError("Can't read an image from the input")
                print("Can't read an image from the input, reconnect..")
                cap = cv2.VideoCapture(rtspUrl)
                ret, frame = cap.read()
                if ret is False or frame is None:
                    print(' reconnect the camera is failure. ')
                    time.sleep(10)
                    continue
        except Exception as e:
            print('error.' + str(e))
            print(f' connect the camera failure. {rtspUrl}')
            time.sleep(30)
            continue
        # key = cv2.waitKey(100)
        if (imageCount % 3 != 0):
            continue
        frame = imutils.resize(frame, width=width)
        # args.no_show= True
        if not args.no_show:
            # 畫出兩個框框
            frame = drawMyArea(frame)
            '''new'''
            # continueTimes+=1
            # cv2.setMouseCallback('image', draw_triggerLine, frame)
            # 設定opencv的滑鼠事件偵測函式
            cv2.setMouseCallback('Detection Results', draw_triggerArea, frame)

            key = cv2.waitKey(100)

            ESC_KEY = 27
            # Quit.
            if key in {ord('q'), ord('Q'), ESC_KEY}:
                break
            # presenter.handleKey(key)

        if detector_pipeline.is_ready():
            # Get new image/frame
            start_time = perf_counter()
            if args.output_resolution:
                output_resolution = output_transform.new_resolution
            else:
                output_resolution = (frame.shape[1], frame.shape[0])
            if next_frame_id == 0:
                output_transform = models.OutputTransform(frame.shape[:2], args.output_resolution)
                presenter = monitors.Presenter(args.utilization_monitors, 55,
                                               (round(output_resolution[0] / 4), round(output_resolution[1] / 8)))
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                # if args.output and not video_writer.open(args.output, fourcc,
                #                                          10, output_resolution):
                    # if args.output and not video_writer.open(args.output, cv2.VideoWriter_fourcc(*'MJPG'),
                    #                                      cap.get(cv2.CAP_PROP_FPS), output_resolution):
                    # raise RuntimeError("Can't open video writer")
            # Submit for inference
            detector_pipeline.submit_data(frame, next_frame_id, {'frame': frame, 'start_time': start_time})
            next_frame_id += 1

        else:
            # Wait for empty request
            detector_pipeline.await_any()

        if detector_pipeline.callback_exceptions:
            # print('aaaaaaaaaaaaaaaaa')
            raise detector_pipeline.callback_exceptions[0]
        # Process all completed requests
        results = detector_pipeline.get_result(next_frame_id_to_show)
        if results:
            objects, frame_meta = results
            frame = frame_meta['frame']
            start_time = frame_meta['start_time']

            if len(objects) and args.raw_output_message:
                print_raw_results(frame.shape[:2], objects, model.labels, args.prob_threshold)

            presenter.drawGraphs(frame)

            '''draw object '''
            frame, boxesNum = draw_detections(frame, objects, palette, model.labels, args.prob_threshold, output_transform
                                    )
            metrics.update(start_time, frame)
            if args.output_resolution:
                output_resolution = output_transform.new_resolution
            else:
                output_resolution = (frame.shape[1], frame.shape[0])
            if args.output:
                recordFileProcess(frame, boxesNum,video_writer,output_resolution)

            # if video_writer.isOpened() and (args.output_limit <= 0 or next_frame_id_to_show <= args.output_limit - 1):
            #     if isPeopleActivate > 0:
            #         video_writer.write(frame)
            next_frame_id_to_show += 1

        if not args.no_show:
            cv2.imshow('Detection Results', frame)

        del frame
        gc.collect()

        if next_send_mqtt_time <= time.time():
            next_send_mqtt_time = int(time.time()) + mqtt_run_cycle_sec

            processB = Process(target=job, args=(parkingSpaces,))
            processB.start()
            for pid in posId:
                # draw.text((1,1), f'{parkingSpaces[pid].inDirCount}_{parkingSpaces[pid].outDirCount}',
                #           fill=color, font=font, outline=color)
                parkingSpaces[pid].isOcc = 0
                # parkingSpaces[pid].indexIds = []
                parkingSpaces[pid].exitCount = 0
                parkingSpaces[pid].insideCount = 0
                parkingSpaces[pid].insideCount_last = 0
                parkingSpaces[pid].inDirCount = 0
                parkingSpaces[pid].outDirCount = 0

    detector_pipeline.await_all()
    # Process completed requests
    for next_frame_id_to_show in range(next_frame_id_to_show, next_frame_id):
        results = detector_pipeline.get_result(next_frame_id_to_show)
        while results is None:
            results = detector_pipeline.get_result(next_frame_id_to_show)
        objects, frame_meta = results
        frame = frame_meta['frame']
        start_time = frame_meta['start_time']

        if len(objects) and args.raw_output_message:
            print_raw_results(frame.shape[:2], objects, model.labels, args.prob_threshold)

        presenter.drawGraphs(frame)
        frame = draw_detections(frame, objects, palette, model.labels, args.prob_threshold, output_transform)
        metrics.update(start_time, frame)

        # if video_writer.isOpened():  # and (args.output_limit <= 0 or next_frame_id_to_show <= args.output_limit - 1):
        #     if isPeopleActivate > 0:
        #         video_writer.write(frame)

        if not args.no_show:
            cv2.imshow('Detection Results', frame)
            key = cv2.waitKey(1)

            ESC_KEY = 27
            # Quit.
            if key in {ord('q'), ord('Q'), ESC_KEY}:
                break
            presenter.handleKey(key)
    if video_writer is not None:
        video_writer.release()
    metrics.print_total()
    print(presenter.reportMeans())

    # processB.terminate()
    print('all process is stoped.')

def tryCatchException(e:Exception):
    print('error.' + str(e))
    error_class = e.__class__.__name__  # 取得錯誤類型
    detail = e.args[0]  # 取得詳細內容
    cl, exc, tb = sys.exc_info()  # 取得Call Stack
    lastCallStack = traceback.extract_tb(tb)[-1]  # 取得Call Stack的最後一筆資料
    fileName = lastCallStack[0]  # 取得發生的檔案名稱
    lineNum = lastCallStack[1]  # 取得發生的行號
    funcName = lastCallStack[2]  # 取得發生的函數名稱
    errMsg = "File \"{}\", line {}, in {}: [{}] {}".format(fileName, lineNum, funcName, error_class, detail)
    print(errMsg)
    print(traceback.format_exc())


#有車時錄，沒車就不錄
def recordFileProcess(frame, boxesNum:int,video_writer:cv2.VideoWriter,output_resolution):
    global recordFileProcess_Count,recordFileProcess_CountMa
    try:
        if boxesNum ==0:
            recordFileProcess_Count+=1
            if recordFileProcess_Count>=recordFileProcess_CountMax:
                if video_writer.isOpened():
                    video_writer.release()
                    print('the video realease()   ')
            else:
                video_writer.write(frame)
        elif boxesNum>0:
            recordFileProcess_Count=0
            if video_writer.isOpened() is False:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                curr_dt = datetime.now()  # 2021-08-25 15:04:33.794484
                nowStr = curr_dt.strftime('%Y-%m-%d_%H-%M-%S')
                recordFile = f'record_{nowStr}.avi'
                video_writer.open(recordFile, fourcc,
                                                 5, output_resolution)
            video_writer.write(frame)
    except Exception as e:
        tryCatchException(e)


if __name__ == '__main__':
    sys.exit(main() or 0)
