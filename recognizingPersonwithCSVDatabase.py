from collections import Iterable
import numpy as np
import imutils
import pickle
import time
import cv2
import csv
from datetime import datetime
import pandas as pd
import collections

def flatten(lis):
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:
            yield item

timeout = time.time()+ 60
count_num=0
embeddingFile = "output/embeddings.pickle"
embeddingModel = "openface_nn4.small2.v1.t7"
recognizerFile = "output/recognizer.pickle"
labelEncFile = "output/le.pickle"
conf = 0.5

print("[INFO] loading face detector...")
prototxt = "model/deploy.prototxt"
model = "model/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt, model)

print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(embeddingModel)

recognizer = pickle.loads(open(recognizerFile, "rb").read())
le = pickle.loads(open(labelEncFile, "rb").read())

Roll_Number = ""
box = []
print("[INFO] starting video stream...")
cam = cv2.VideoCapture(0)
time.sleep(2.0)
name_list = []
rollno_list = []
accuracy_list =[]
time_list = []
while True:
    _, frame = cam.read()
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    detector.setInput(imageBlob)
    detections = detector.forward()

    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > conf:

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            if fW < 20 or fH < 20:
                continue

            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]

            with open('student.csv', 'r') as csvFile:
                reader = csv.reader(csvFile)
                for row in reader:
                    box = np.append(box, row)
                    name = str(name)
                    if name in row:
                        person = str(row)
                        count_num += 1
                        print(name)
                        print(proba*100)

                listString = str(box)
                if name in listString:
                    singleList = list(flatten(box))
                    listlen = len(singleList)
                    Index = singleList.index(name)
                    name = singleList[Index]
                    Roll_Number = singleList[Index + 1]
                    print(Roll_Number)
                    now = datetime.now()
                    dtString = now.strftime('%H:%M:%S')
                    if (round(proba*100))>70:
                        name_list.append(name)
                        rollno_list.append(Roll_Number)
                        time_list.append(dtString)
                        accuracy_list.append((round(proba*100)))

            text = "{} : {} : {:.2f}%".format(name, Roll_Number, proba * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or time.time()>timeout:
        break
    time.sleep(0.3)
dict_mark = {'rollno': rollno_list, 'name': name_list, 'time': time_list, 'accuracy': accuracy_list}
counter = collections.Counter(rollno_list)
pre_rollno = []
for nam in counter:
    print(f"{nam}  ->  {round((counter[nam]/len(name_list))*100)}")
    if (round((counter[nam]/len(name_list))*100))> 10:
        pre_rollno.append(int(nam))

df = pd.DataFrame(dict_mark)
df.to_csv('file2.csv', index=False)
student_data = pd.read_csv("student.csv")
student_data = student_data.to_dict(orient="records")

now2 = datetime.now()
date_str = now2.strftime("%d-%m-%Y")
student_data = pd.read_csv("student.csv")
student_data = student_data.to_dict(orient="records")

namePresent = []
roll_absent = []
name_absent = []
for dic in student_data:
    if dic["Roll_No"] in pre_rollno:
        namePresent.append(dic["name"])
    else:
        name_absent.append(dic["name"])
        roll_absent.append(dic["Roll_No"])
dict_present = {"Roll_no": pre_rollno, "name": namePresent, "attendance": ["P" for i in pre_rollno]}
df2 = pd.DataFrame(dict_present)
dict_absent = {"Roll_no": roll_absent, "name": name_absent, "attendance": "A"}
df3 = pd.DataFrame(dict_absent)
df4 = pd.concat([df2, df3], ignore_index=True)
df4.reset_index()
df4 = df4.sort_values(by=['Roll_no'])
df4.to_csv(f"{date_str}.csv", index=False)
#print(count_num)
print(df4)
cam.release()
cv2.destroyAllWindows()