# Live Detection of DHL Cars using finetuned YOLOv11 on a custom dataset and Pushover
### Crux
During my Rehab, I had the problem that whenever a DHL Driver knocked at my door, I was too slow to get up and open the door for them. This meant that I could never take reception of my packages on time. To overcome this crux, I was wondering if I could be informed about the arrival of a DHL Truck, way before they knock at my door. 
### Solution
To do this, I trained a **YOLOv11** model on a **Custom Dataset** containing DHL Logos and used the **Pushover API** to send notifications to my phone in real time.</br>
The finetuned YOLOv11m model with 20,033,116 parameters was trained over 100 Epochs on the custom dataset and reached an **mAP50-95** of **0.6** across all classes.</br>
Detections of a DHL Logo are logged with the corresponding time of detection in a seperate log file, while the images containing the bounding boxes are saved as well. 
### API
To make this a usable detection system, I used the Pushover API because it allows for seamless interaction between smartphones and python applications. Everytime a DHL Logo gets detected, a message containing the time of the detection together with the corresponding, annotated image gets directly sent to my phone. This program can then be used in combination with a seperate camera for even better object detection performance.</br>
Below is an Image that portrays the result of my system.


*Since I can only upload files upto 25MB and the model is 40.8MB big, the finetuned model can only be accessed through contacting me directly*.

<img src="https://github.com/fylexx/Projects/blob/main/Live_DHL_Detection_YOLO/detection_2025-07-17%2019-40-31.jpg" width="500" height="700"> </br>
**Dataset** for finetuning: https://universe.roboflow.com/logo-8yuyu/dhl-1pl8g
