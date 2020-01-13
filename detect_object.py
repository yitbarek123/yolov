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
        self.labels=open("yolo-coco/coco.names").read().split("\n")
    
    def load_model(self):
        net=cv2.dnn.readNetFromDarknet(self.configpath,self.weightpath)
        return net
    
    def process_image(self,image):
        (H,W)=image.shape[:2]
        return H,W
    
    def get_outputs(self,image):
        net=self.load_model()
        layer_name=net.getLayerNames()
        layer_name=[layer_name[i[0]-1] for i in net.getUnconnectedOutLayers()]
        blob=cv2.dnn.blobFromImage(image,1/255.0,(416,416),swapRB=True,crop=False)
        net.setInput(blob)
        output=net.forward(layer_name)
        return output
    
    def forward(self,image):
        outputs=self.get_outputs(image)
        H,W=self.process_image(image)
        boxes=[]
        confidences=[]
        classIDs=[]
        for output in outputs:
            for detection in output:
                scores=detection[5:]
                classID=np.argmax(scores)
                confidence=scores[classID]
                if confidence>0.5:
                    box=detection[0:4]*np.array([W,H,W,H])
                    (centerX,centerY,width,height)=box.astype("int")
                    x=int(centerX-(width/2))
                    y=int(centerY-(height/2))
                    boxes.append([x,y,int(width),int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
        idxs=cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.3)

        return boxes,confidences,classIDs,idxs
    
    def run_app(self,image):
        image=cv2.imread(image)
        np.random.seed(42)
        COLORS=np.random.randint(0,255,size=(len(self.labels),3),dtype="uint8")
        self.load_model()
        boxes,confidences,classIDs,idxs=self.forward(image)
        if len(idxs)>0:
            for i in idxs.flatten():
                (x,y)=(boxes[i][0],boxes[i][1])
                (w,h)=(boxes[i][2],boxes[i][3])
                color=[int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(image,(x,y,),(x+w,y+h),color,2)
                text="{}:{:.4f}".format(self.labels[classIDs[i]],confidences[i])
                cv2.putText(image,text,(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)
        cv2.imshow("Image",image)
        cv2.waitKey(0)

if __name__=="__main__":
    yolo=YOLO()
    yolo.run_app("images/soccer.jpg")