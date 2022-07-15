# -*- coding: utf-8 -*-
import locale
import cv2_ext
import cv2

locale.setlocale(locale.LC_ALL, '')

thres = 0.65  # Threshold to detect object

cap = cv2.VideoCapture("video.mp4")
cap.set(3, 1280)
cap.set(4, 720)
cap.set(10, 70)

classNames = []
classFile = 'veri_seti.names'

with open(classFile,'rt') as f:
    classNames=[line.rstrip() for line in f]

# with open(classFile, 'rt') as f:
#     classNames = f.read().rstrip('n').split('n')

configPath = 'ayardosyasi.pbtxt'
weightsPath = 'etkilesim_grafikler.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    print(classIds, bbox)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=1)
            cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 5, box[1] + 15),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 170, box[1] + 15),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
            try:
                if (classIds == 17):
                    print("vega gorundu")
            except ValueError:
                print("Çoklu veri girişi tespit edildi!")
    cv2.imshow("Tanima penceresi",img)
    key = cv2.waitKey(1) & 0xFF
    # Burada "Q" tuşu ile quit komutu verilir ve program kapanır.
    if key == ord("q"):
        break
    #cv2.waitKey(1)

