import numpy as np
import cv2
import os
import time

class YOLO():
    def __init__(self):
        self.confidence=0.5
        self.threshold=0.3
        self.weightpath="yolo-coco/yolov3.weights"
        self.configpath="yolo-coco/yolov3.cfg"
        self.labels=open("yolo-coco/coco.names").read().strip().split("\n")

    def load_model(self):
        net=cv2.dnn.readNetFromDarknet(self.configpath,self.weightpath)
        return net
    
    def get_image_hw(self, image):
        (H,W)=image.shape[:2]
        return H ,W
    
    def get_output_layer(self,net):
        layer_name=net.getLayerNames()
        layer_name=[layer_name[i[0]-1] for i in net.getUnconnectedOutLayers()]
        return layer_name
    
    def forward(self,image):
        net=self.load_model()
        blob=cv2.dnn.blobFromImage(image,1/255.0,(416,416),swapRB=True,crop=False)
        net.setInput(blob)
        start=time.time()
        layer_name=self.get_output_layer(net)
        layerOutputs=net.forward(layer_name)
        end=time.time()
        output=layerOutputs
        return output
    
    def bounding_box(self,images):
        boxes=[]
        confidences=[]
        classIDs=[]
        outputs=self.forward(images)
        H,W=self.get_image_hw(images)
        for output in outputs:
            for detection in output:
                scores=detection[5:]
                classID=np.argmax(scores)
                confidence=scores[classID]
                if confidence>self.confidence:
                    box=detection[0:4]*np.array([W,H,W,H])
                    (centerX,centerY,width,height)=box.astype("int")
                    x=int(centerX-(width/2))
                    y=int(centerY-(height/2))
                    boxes.append([x,y,int(width),int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
        idxs=cv2.dnn.NMSBoxes(boxes,confidences,self.confidence,self.threshold)
        return idxs, confidences, boxes,classIDs            

    def run_app(self, image):
        image=cv2.imread(image)
        idxs,confidences,boxes,classIDs=self.bounding_box(image)
        np.random.seed(42)
        colors=np.random.randint(0,255,size=(len(self.labels),3),dtype="uint8")   

        if len(idxs)>0:
            for i in idxs.flatten():
                if self.labels[classIDs[i]]=="person":
                    (x,y)=(boxes[i][0],boxes[i][1])
                    (w,h)=(boxes[i][2],boxes[i][3])
                    color=[int(c) for c in colors[classIDs[i]]]
                    cv2.rectangle(image,(x,y,),(x+w,y+h),color,2)
                    text="{}:{:.4f}".format(self.labels[classIDs[i]],confidences[i])
                    cv2.putText(image,text,(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)
        cv2.imshow("Image",image)
        cv2.waitKey(0)

if __name__=='__main__':
    yolo=YOLO()
    yolo.run_app("images/soccer.jpg")