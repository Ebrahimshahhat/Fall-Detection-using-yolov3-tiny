import cv2
import numpy as np
import cvzone
import time
import requests

#Pushover Api method

def send_pushover_notification(user_key, app_token, message):
    url = 'https://api.pushover.net/1/messages.json'
    data = {
        'token': app_token,
        'user': user_key,
        'message': message
    }
    files = {'attachment': open(image_path, 'rb')} if image_path else None
    response = requests.post(url, data=data,files=files)
    if response.status_code == 200:
        print("Notification sent successfully.")
    else:
        print("Failed to send notification. Status code:", response.status_code)
#Api cred>> replace them with yours
user_key = 'User_key'
app_token = 'Application_token'
message = "Someone Fell , Help'em "        
#model insertion
net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
 
#Testing Video
cap = cv2.VideoCapture("fall.mp4")
font = cv2.FONT_HERSHEY_PLAIN
starting_time= time.time()
frame_id = 0
colors= np.random.uniform(0,255,size=(len(classes),3))

#main detection loop
while True:
    ret,frame= cap.read() 
    if not ret:
        break;
    frame_id+=1
    
    height,width,channels = frame.shape

    blob = cv2.dnn.blobFromImage(frame,0.00392,(320,320),(0,0,0),True,crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids=[]
    confidences=[]
    boxes=[]
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                
                
                center_x= int(detection[0]*width)
                center_y= int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                
                x=int(center_x - w/2)
                y=int(center_y - h/2)
                
                if classes[class_id] == 'person':
                    height = y + h - y
                    width = x + w - x
                    threshold = height - width

                    cvzone.cornerRect(frame, [x, y, w, h], l=15, rt=2,colorR=(0, 255, 0))
                    cvzone.putTextRect(frame, f'{classes[class_id]} {round(confidence*100)}%', [x + 8, y - 12], thickness=1,
                                       scale=1)

                    if threshold < 0:
                        cv2.putText(frame, 'Fall Detected', (x, y - 32), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                                    cv2.LINE_AA)
                        cvzone.cornerRect(frame, [x, y, w, h], l=15, rt=2,colorR=(0, 0, 255))
                        if frame_id%5 ==0:
                            image_path = "detected_fall.png"
                            cv2.imwrite(image_path, frame)
                            send_pushover_notification(user_key, app_token, message)

                label = str(classes[class_id])
                color = colors[class_id]


    elapsed_time = time.time() - starting_time
    fps=frame_id/elapsed_time
    cv2.putText(frame,"FPS:"+str(round(fps,2)),(10,50),font,2,(255,255,0),1)
    
    cv2.imshow("Image",frame)
    key = cv2.waitKey(1) 
    
    if key == 27: #esc key stops the process
        break;
    
cap.release()    
cv2.destroyAllWindows()  
