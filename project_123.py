import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os,ssl,time
if (not os.environ.get("PYTHONHTTPSVERIFY","")and
getattr(ssl,"_create_unverified_context",None)):
  ssl._create_default_https_context=ssl._create_unverified_context
#X,y=fetch_openml("mnist_784",version=1,return_X_y=True)
#print(pd.Series(y).value_counts())
X=np.load('image.npz')["arr_0"]
y=pd.read_csv("labels.csv")["labels"]
classes=["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
nclasses=len(classes)
print(X)
print(nclasses)
Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,y,random_state=9,train_size=7500,test_size=2500)
Xtrainscaled=Xtrain/255.0
Xtestscaled=Xtest/255.0
clf=LogisticRegression(solver="saga",multi_class="multinomial").fit(Xtrainscaled,Ytrain)
yprediction=clf.predict(Xtestscaled)
accuracy=accuracy_score(Ytest,yprediction)
print(accuracy)
cap=cv2.VideoCapture(0) 
while(True):
    try:
        ret,frame=cap.read()
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        height,width=gray.shape
        upper_left=(int(width/2-56),int(height/2-56))
        bottom_right=(int(width/2+56),int(height/2+56))
        cv2.rectangle(gray,upper_left,bottom_right,(0,255,0),2)
        roi=gray[upper_left[1]:bottom_right[1],upper_left[0]:bottom_right[0]]
        img_pil=Image.fromarray(roi)
        img_bw=img_pil.convert("L")
        img_bw_resized=img_bw.resize((28,28),Image.ANTIALIAS)
        img_bw_resized_inverted=PIL.ImageOps.invert(img_bw_resized)
        pixel_filter=20                                    
        minimum_pixel=np.percentile(img_bw_resized_inverted,pixel_filter)
        img_inverted_scaled=np.clip(img_bw_resized_inverted-minimum_pixel,0,255)
        maximum_pixel=np.max(img_bw_resized_inverted)
        img_bw_resized_scaled=np.asarray(img_inverted_scaled)/maximum_pixel
        test_sample=np.array(img_bw_resized_scaled).reshape(1,784)
        test_prediction=clf.predict(test_sample)
        print("predicted class is ",test_prediction)
        cv2.imshow("frame",gray)
        if cv2.waitKey(1) & 0xFF==ord("q"):
            break
    except Exception as e:
        pass
cap.release()
cv2.destroyAllWindows()