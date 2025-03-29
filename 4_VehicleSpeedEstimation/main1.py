import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import*
import time
from math import dist
import matplotlib.pyplot as plt

model=YOLO('yolov8s.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap=cv2.VideoCapture('veh2.mp4')


my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
#print(class_list)

count=0

tracker=Tracker()

offset=6                          # Jaise hi gadi touch hogi waise hi detect karna hai -> use offset

cy1=322
cy2=368

veh_down = {}
counter = []
veh_down_time = {}

veh_up = {}
counter1 = []
veh_up_time = {}

time_list_down = []
distance_list_down = []

time_list_up = []
distance_list_up = []

while True:    
    ret,frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(1020,500))
   

    results=model.predict(frame)
 #   print(results)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
#    print(px)
    list=[]
             
    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        if 'car' in c:
            list.append([x1,y1,x2,y2])
    bbox_id=tracker.update(list)
    for bbox in bbox_id:
        x3,y3,x4,y4,id=bbox
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2
        
        cv2.rectangle(frame, (x3,y3), (x4,y4), (0,0,255), 2)
        
        ## Going down
        if cy1 < (cy + offset) and cy1 > (cy - offset):
            veh_down[id] = time.time()          # jab vehicle first line ko touch kar rha hai tab uski id ke corresponding uska time save kar le rahe hai
        if id in veh_down:                      # unhi vehicles ko count karenge jo humari dict mein hai (line 1 touch kari hai unhone)
            if cy2 < (cy + offset) and cy2 > (cy - offset):  # now if the vehicle touches the second line (line 2)
                elapsed_time = time.time() - veh_down[id]     # uska current time note kar liya jab vo second line ko touch kiya and line 1 ka time subtract kar diya
                if counter.count(id) == 0:
                    counter.append(id)
                    distance = 10    # in meters
                    a_speed_ms = distance / elapsed_time
                    a_speed_kh = a_speed_ms * 3.6
                    time_list_down.append(elapsed_time)
                    distance_list_down.append(distance)
                    cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                    cv2.putText(frame,str(id),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,0.6,(255,255,255),1)
                    cv2.putText(frame,str(int(a_speed_kh))+'Km/h',(x4,y4),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
        
        ## Going up
        if cy2 < (cy + offset) and cy2 > (cy - offset):
            veh_up[id] = time.time()
        if id in veh_up:                      # unhi vehicles ko count karenge jo humari dict mein hai
            if cy1 < (cy + offset) and cy1 > (cy - offset):
                elapsed_time_1 = time.time() - veh_up[id]
                if counter1.count(id) == 0:
                    counter1.append(id)
                    distance1 = 10
                    a_speed_ms1 = distance1 / elapsed_time_1
                    a_speed_kh1 = a_speed_ms1 * 3.6
                    time_list_up.append(elapsed_time_1)
                    distance_list_up.append(distance1)
                    cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                    cv2.putText(frame,str(id),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,0.6,(0,255,255),1)
                    cv2.putText(frame,str(int(a_speed_kh1))+'Km/h',(x4,y4),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
                    
           
    cv2.line(frame,(274,cy1),(814,cy1),(255,255,255),1)
    cv2.putText(frame,('LINE 1'),(277,320),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
    cv2.line(frame,(177,cy2),(927,cy2),(255,255,255),1)
    cv2.putText(frame,('LINE 2'),(182,367),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
    
    length_down = (len(counter))
    cv2.putText(frame,('VehicleDown:')+str(length_down),(60,40),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
    
    length_up = (len(counter1))
    cv2.putText(frame,('VehicleUp:')+str(length_up),(60,130),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
    
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break

plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(time_list_down, distance_list_down, marker='o')
plt.title('Distance-Time Graph for Vehicles Going Down')
plt.xlabel('Time (seconds)')
plt.ylabel('Distance (meters)')
plt.subplot(2, 1, 2)
plt.plot(time_list_up, distance_list_up, marker='o')
plt.title('Distance-Time Graph for Vehicles Going Up')
plt.xlabel('Time (seconds)')
plt.ylabel('Distance (meters)')
plt.tight_layout()
plt.show()

cap.release()
cv2.destroyAllWindows()