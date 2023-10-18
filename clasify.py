import numpy as np
import cv2

IMG_HEIGHT = 64
IMG_WIDTH = 64

NUM_CLASSES = 14
CLASS_LABELS = ['Abuse','Arrest','Arson','Assault','Burglary','Explosion','Fighting',"Normal",'RoadAccidents','Robbery','Shooting','Shoplifting','Stealing','Vandalism']

def freq(a):
  from collections import Counter
  count = Counter(a)
  # Most frequent value
  res = count.most_common(1)
  # Display result
  return res[0][0]


def classify(model,frames):
  print(frames.shape)
  # frames = build_transforms()(frames)
  preded = []
  for f in frames:
    # print(f.shape)
    f = cv2.resize(f, (IMG_HEIGHT, IMG_WIDTH))
    processed_image = f.reshape( -1,IMG_HEIGHT, IMG_WIDTH,3) 
    # print(processed_image.shape)
    pred=model.predict(processed_image)
    # print("Prediction:")
    # print(CLASS_LABELS[pred.argmax()])
    preded.append(CLASS_LABELS[pred.argmax()])
  print(freq(preded))
  print(preded)


    
    



    
    
