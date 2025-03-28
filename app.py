import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO

model=YOLO('yolov8s.pt')                        # Creating an instance of the YOLO model
def RGB(event, x, y, flags, param):             # function to handle mouse events. Prints the BGR colour values when the mouse moves over the image. These parameters are automatically provided by OpenCV when a mouse event occurs.
    if event == cv2.EVENT_MOUSEMOVE:            # this line checks if the event is a mouse event or not, represents the event type when the mouse cursor moves over the image
        colorsBGR = [x, y]                      # when mouse move event occurs, x and y coordinates of the cursor are extracted and stored in list colorsBGR
        print(colorsBGR)                        # prints the list containing x and y coordinates of the mouse cursor. Printed values represent the position of the cursor within the image.
        
cv2.namedWindow('RGB')                          # creating a named window called RGB
cv2.setMouseCallback('RGB', RGB)                # associate the 'RGB' function to the mouse events in the 'RGB' window. Prints the x and y coordinates of the mouse cursor when it moves over the image displayed in the 'RGB' window.

cap=cv2.VideoCapture('video.mp4')           # read the frames from the video

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")                   # splitting content of file into list of classes using the newline character
print(class_list)                               # Prints the list of classes to the console

count=0                                         # to keep track of the number of frames processed
while True:                                     # Starting infinite loop for processing video frames
    
    ret,frame = cap.read()                      # ret is a boolean value indicating whether the frame has been actually read or not, if ret is true, then frame contains the image data of the current frame. If ret is false, there are no more frames to be read. 
    frame=cv2.resize(frame,(1020,500))          # resize the current frame to a specified width 1020 and height 500
    
    if not ret:
        break                                   # if ret is false it means either there is a problem to read the frames or the video has reached its end.
    count += 1
    if count % 3 != 0:
        continue                                # this code ensures that only every third frame is processed in the loop. The purpose of skipping frames is often to reduce computational load and processing time. By processing only every third frame, the code can still capture the overall motion in the video while potentially reducing the amount of processing required, which can be beneficial in real-time applications or when dealing with large video datasets.

    #results=model.predict(frame, show = True)   # makes predictions on the model instantiated above taking current frame as an input parameter and show = True to visualize the predicted results  
           
    #cv2.imshow("RGB", frame)                    # display the images in the "RGB" window for current frame
    #if cv2.waitKey(1)&0xFF==27:                 # cv2.waitKey is used to wait for a key event 1 millisecond here. 0xFF is bitwise AND to extract last 8 bits -> mask the ASCII value of the key pressed as only last 8 bits might be relevant, ==27 maps to ASCII value of the 'Esc' key
    #   break                                   # if above condition is true meaning escape was pressed, the loop breaks
    results = model.predict(frame)
    #print(results)                             # This gives a 6 column matrix, the first 4 columns are for the bounding box, rectangular coordinates. 5th column is the confidence level for object detection and 6th column is the class
    box=results[0].boxes.xyxy                     # used to extract the bounding box coordinates from the results of the YOLO model's prediction. results[0]: Accesses the first element of the results. .boxes: Accesses the detection boxes within the prediction results. .xyxy: Accesses the bounding box coordinates in the format [x1, y1, x2, y2], where (x1, y1) are the coordinates of the top-left corner, and (x2, y2) are the coordinates of the bottom-right corner. So, a now contains the bounding box coordinates of detected objects in the current frame.
    #print(a)
    final_box = pd.DataFrame(box).astype("float")               # to convert the bounding box coordinates to pandas dataframe and sets the data type to float
    print("Printing bounding box coordinates: \n", final_box)
    print("\n")
    
    confidences = results[0].boxes.conf                           # extracts the confidence scores from YOLO prediction model 
    final_confidences = pd.DataFrame(confidences)                 # converts them to a panda dataframe
    print("Printing confidences: \n", final_confidences)
    print("\n")
    
    classes = results[0].boxes.cls                                # Extracts the class labels
    final_classes = pd.DataFrame(classes).astype(int)             # Convert the class labels to a DataFrame and set the data type to int
    print("Printing classes: \n", final_classes)
    
    for index, row in final_box.iterrows():                       # Iterate through the rows of the final_box DataFrame
        x1 = int(row[0])                                          # x1,y1,x2,y2 are the coordinates for the object
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)     # to make a rectangle around the objects moving in the video (0,0,255) -> red, and 2 is the thickness of the rectangle box

        class_index = final_classes.iloc[index, 0]                  # Get the class index for the current object. .iloc indexer is used to select rows and columns by integer position. In this case, it is used to locate the value at the specified row (index) and column (0). [index, 0] -> index represents the current row in the iteration of the DataFrame final_box. 0 is the column index. Since final_classes is a DataFrame with a single column containing class labels, 0 refers to that single column.
        object_class = class_list[class_index]                      # class_list: This is a Python list containing the names of different classes. class_index: This variable holds the class index obtained from the previous line (class_index = final_classes.iloc[index, 0]). It represents the predicted class index for the currently processed object. class_list[class_index]: This expression is using the class_index to access the corresponding class label from the class_list. It retrieves the class name of the detected object based on the predicted class index.
        cv2.putText(frame, str(object_class), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)  # current frame from the video on which you want to draw the text, text that you want to put on the image, starting point coordinates where you want to put the text, top-left corner, font type, font scale factor for font size, color of the text -> blue for(255,0,0), thickness of the lines used to draw the text
        
    
    cv2.imshow("RGB", frame)                                        # display the images in the "RGB" window for current frame
    if cv2.waitKey(1)&0xFF==27:                                     # cv2.waitKey is used to wait for a key event 1 millisecond here. 0xFF is bitwise AND to extract last 8 bits -> mask the ASCII value of the key pressed as only last 8 bits might be relevant, ==27 maps to ASCII value of the 'Esc' key
        break                                                       # if above condition is true meaning escape was pressed, the loop breaks

cap.release()                                   # releases the video file, making it free for other processes, and is still not in use
cv2.destroyAllWindows()                         # closes all OpenCV windows
