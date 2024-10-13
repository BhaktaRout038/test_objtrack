##**********************************************V[1.0.0.0]*****************************************************
# it only counts the vehicle

# import cv2
# import numpy as np

# min_contour_width=40  #40
# min_contour_height=40  #40
# offset=10       #10
# line_height=550 #550
# matches =[]
# cars=0
# def get_centroid(x, y, w, h):
#     x1 = int(w / 2)
#     y1 = int(h / 2)

#     cx = x + x1
#     cy = y + y1
#     return cx,cy
#     #return (cx, cy)
        

# cap = cv2.VideoCapture('test3.mp4')




# cap.set(3,1920)
# cap.set(4,1080)

# if cap.isOpened():
#     ret,frame1 = cap.read()
# else:
#     ret = False
# ret,frame1 = cap.read()
# ret,frame2 = cap.read()
    
# while ret:
#     d = cv2.absdiff(frame1,frame2)
#     grey = cv2.cvtColor(d,cv2.COLOR_BGR2GRAY)
#     #blur = cv2.GaussianBlur(grey,(5,5),0)
#     blur = cv2.GaussianBlur(grey,(5,5),0)
#     #ret , th = cv2.threshold(blur,20,255,cv2.THRESH_BINARY)
#     ret , th = cv2.threshold(blur,20,255,cv2.THRESH_BINARY)
#     dilated = cv2.dilate(th,np.ones((3,3)))
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

#         # Fill any small holes
#     closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel) 
#     contours,h = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#     for(i,c) in enumerate(contours):
#         (x,y,w,h) = cv2.boundingRect(c)
#         contour_valid = (w >= min_contour_width) and (
#             h >= min_contour_height)

#         if not contour_valid:
#             continue
#         cv2.rectangle(frame1,(x-10,y-10),(x+w+10,y+h+10),(255,0,0),2)
        
#         cv2.line(frame1, (0, line_height), (1200, line_height), (0,255,0), 2)
#         centroid = get_centroid(x, y, w, h)
#         matches.append(centroid)
#         cv2.circle(frame1,centroid, 5, (0,255,0), -1)
#         cx,cy= get_centroid(x, y, w, h)
#         for (x,y) in matches:
#             if y<(line_height+offset) and y>(line_height-offset):
#                 cars=cars+1
#                 matches.remove((x,y))
#                 print(cars)
                
#     cv2.putText(frame1, "Total Detected: " + str(cars), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1,
#                     (0, 170, 0), 2)

#     cv2.putText(frame1, "Testing", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
#                     (255, 170, 0), 2)
    
    
    
#     #cv2.drawContours(frame1,contours,-1,(0,0,255),2)


#     cv2.imshow("Original" , frame1)
#     #cv2.imshow("Difference" , th)
#     if cv2.waitKey(1) == 27:
#         break
#     frame1 = frame2
#     ret , frame2 = cap.read()
# #print(matches)    
# cv2.destroyAllWindows()
# cap.release()


##**********************************************V[1.0.0.1]*****************************************************
#it counts vehicle & show vehicle category


import cv2
from ultralytics import YOLO

min_contour_width = 40
min_contour_height = 40
offset = 10
line_height = 550
cars = 0
trucks = 0
buses = 0
motorbikes = 0

# Load YOLOv8 model (you can use 'yolov8n.pt', 'yolov8s.pt', etc. based on your preference)
model = YOLO('yolov8n.pt')  # Use appropriate YOLOv8 model file

cap = cv2.VideoCapture('test3.mp4')
cap.set(3, 1920)
cap.set(4, 1080)

if cap.isOpened():
    ret, frame1 = cap.read()
else:
    ret = False

def get_centroid(x, y, w, h):
    cx = int(x + w / 2)
    cy = int(y + h / 2)
    return cx, cy

while ret:
    # Run YOLOv8 inference
    results = model(frame1)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            w = x2 - x1
            h = y2 - y1
            confidence = box.conf[0]  # Confidence score
            class_id = int(box.cls[0])  # Class ID
            label = model.names[class_id]  # Class name from YOLOv8

            if w >= min_contour_width and h >= min_contour_height:
                centroid = get_centroid(x1, y1, w, h)
                cv2.rectangle(frame1, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame1, f'{label} {int(confidence * 100)}%', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.circle(frame1, centroid, 5, (0, 255, 0), -1)

                cx, cy = centroid
                if line_height - offset < cy < line_height + offset:
                    if label == 'car':
                        cars += 1
                    elif label == 'truck':
                        trucks += 1
                    elif label == 'bus':
                        buses += 1
                    elif label == 'motorbike':
                        motorbikes += 1

    # Display the counts of different vehicle types
    cv2.putText(frame1, f"Cars: {cars}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame1, f"Trucks: {trucks}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame1, f"Buses: {buses}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame1, f"Motorbikes: {motorbikes}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.line(frame1, (0, line_height), (frame1.shape[1], line_height), (0, 255, 0), 2)

    # Show the frame with detections and counts
    cv2.imshow("Vehicle Detection", frame1)

    if cv2.waitKey(1) == 27:  # Press 'ESC' to exit
        break

    ret, frame1 = cap.read()

cap.release()
cv2.destroyAllWindows()








