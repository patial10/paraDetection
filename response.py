import cv2
import numpy as np
import time
import sys
import os
import shutil
import pytesseract
pytesseract.pytesseract.tesseract_cmd=r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def get_response(path_name):
    dic=[]
    print(path_name)
    confidence=0.5
    score_threshold=0.5
    iou_threshold=0.5
    config_path="darknet/cfg/yolov4-custom.cfg"
    #yolo net weights file 
    weight_path="darknet/data/backup/yolov4-custom_last.weights"
    #loading all class labels (objects)
    labels=open("darknet/data/obj.names").read().strip().split("\n")
    colors = np.random.uniform(0, 255, size=(len(labels), 3))
    # print(labels)    
    #load the yolo network
    net = cv2.dnn.readNetFromDarknet(config_path,weight_path)
    #Preparing image now
#     path_name=open(path_name, 'rb').read()
#     path_name="static/uploads/input.jpg"
    image=cv2.imread(path_name)
#     file_name=os.path.basename(path_name)
#     filename,ext=file_name.split(".")
    #Next, we need to normalize,scale,reshape this image to be suitable as an input to the neural network:
    h,w=image.shape[:2]
    #create 4D blob
    blob=cv2.dnn.blobFromImage(image,1/255.0,(416,416),swapRB=True,crop=False)#Thiswillnormalize pixelvaluesrnge from 0 to 1.let see
    print("image.shape",image.shape)
    print("blob.shape",blob.shape)
    # now making some prediction let see:
    #feed this image to neural network to get output:
    #set the blob as input of the network
    net.setInput(blob)
    #get all layers names
    ln=net.getLayerNames()
    print(ln)
    ln=[ln[i[0]-1] for i in net.getUnconnectedOutLayers()]
    print(ln)
    #perform the forward(inference) and  get nn output
    layer_output=net.forward(ln)
    #we need to iterate over the nn outputs and discardany objectthat has the confidence less than confidence parameter 
    #we specified earlier
    font_scale=1
    thickness=1
    boxes,confidences,class_i=[],[],[]
    
    #loop over each of the layer outputs
    for output in layer_output:
        #loop over the each object detections
        for detect in output:
            scores = detect[5:]
    #         print(scores)
            class_idd = np.argmax(scores)
            confs = scores[class_idd]
            if confs > confidence:
                center_x = int(detect[0] * w)
                center_y = int(detect[1] * h)
                wi = int(detect[2] * w)
                hi = int(detect[3] * h)
                x = int(center_x - wi /2)
                y = int(center_y - hi / 2)
                boxes.append([x, y, wi, hi])
                confidences.append(float(confs))
    #             print(class_idd)
                class_i.append(class_idd)
    #     print(len(class_i))
    #with NMS boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold, 0.5)
    j=0
    for i in range(len(boxes)):
        if i in indexes:
    #         print(class_i[i])
            x, y, w, h = boxes[i]
            label = str(labels[class_i[i]])
            color = colors[class_i[i]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, label, (x, y -10),cv2.FONT_HERSHEY_SIMPLEX,1, color,2)
            cropped_img=image[y:y+h, x:x+w]
            cv2.imshow("Image",cropped_img)
            extracted_text=pytesseract.image_to_string(cropped_img)
            print(extracted_text)  
            print(x,y,w,h)

            dic.append({'x':x,'y':y,'w':w,'h':h,'label':label,'paragraph':extracted_text})
            j=j+1
    # Filename 
    filename1 = 'result.jpg'
    k=0
    while True:
        if os.path.isfile("static/detections/"+filename1):              
            print(os.path.isfile("static/detections/"+filename1))
            filename1='result'+str(k)+'.jpg'
            k=k+1
        else:
            break

        
    cv2.imwrite('static/detections/'+filename1, image)
    gg=filename1
    print(gg)
    return gg,dic
    
    
    
    
    
    
    
    
    