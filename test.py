import cv2
from prediction import perform_prediction
from numpy.lib.function_base import copy


import cv2
import time
from clasify import classify
from keras.models import model_from_json
from alert_sys import send_call , send_message


def anomaly_classifier_model():
    json_file = open('./models/classifier.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("./models/classifier_weights.h5")
    return loaded_model

anomaly_classifier = anomaly_classifier_model()

i = 0
clip_length=16
# file_path = r"C:\Users\Bharath\Downloads\test_2.mp4"
file_path = "./test/kill.mp4"
cap = cv2.VideoCapture(file_path)
# cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4")


frames_queue = []
import time
while True:
    # print("REading")
    ret, cv_img = cap.read()
    if ret:
        # cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        cv2.imshow('frame',cv_img)
        if i<clip_length:
            frames_queue.append(cv_img)
            frame = cv_img
            i+=1
        else:
            i=0
            batch = copy(list(reversed(frames_queue)))
            frames_queue = []

            prediction = perform_prediction(batch)

            if (prediction > 0.8):
                classify(model=anomaly_classifier,frames=batch)
                
                frame_shape = batch[0].shape
                height = frame_shape[0]
                width = frame_shape[1]
                fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
                video_writer = cv2.VideoWriter(f"./output/result_camera_id_{str(time.time_ns())}.mp4", fourcc, 10.0, (width,height))

                for frame in batch:
                    video_writer.write(frame)
                video_writer.release()
                print("saved video")

                # if prediction >0.9:
                #     send_call()
                # else:
                #     send_message()

                print("Anamoly DETECTED: " + str(prediction))
            else:
                print("Normal Video " + str(prediction))


        if cv2.waitKey(1) == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()
