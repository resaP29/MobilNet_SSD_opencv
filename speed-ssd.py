import cv2
import time

GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
fonts = cv2.FONT_HERSHEY_COMPLEX

Distance_level = 0
travedDistance = 0
changeDistance = 0
velocity = 0

classNames = { 0: 'background',
    1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
    5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
    10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
    14: 'motorbike', 15: 'person', 16: 'pottedplant',
    17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }

cap = cv2.VideoCapture('1.MOV')
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt', 'MobileNetSSD_deploy.caffemodel')

def averageFinder(valuesList, numberElements):
    sizeOfList = len(valuesList)
    print(sizeOfList)
    lastMostElement = sizeOfList - numberElements
    lastPart = valuesList[lastMostElement:]
    average = sum(lastPart)/(len(lastPart))
    return average

speedList = []
DistanceList = []
averageSpeed = 0
jarakAwal = 0

while True:
    _, frame = cap.read()

    intialTime = time.time()
    frame_resized = cv2.resize(frame, (300, 300))  # resize frame for prediction
    blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
    net.setInput(blob)
    detections = net.forward()
    cols = frame_resized.shape[1]
    rows = frame_resized.shape[0]
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]  # Confidence of prediction
        if confidence > 0.1:  # Filter prediction
            class_id = int(detections[0, 0, i, 1])  # Class label
            if class_id == 14:
                xLeftBottom = int(detections[0, 0, i, 3] * cols)
                yLeftBottom = int(detections[0, 0, i, 4] * rows)
                xRightTop = int(detections[0, 0, i, 5] * cols)
                yRightTop = int(detections[0, 0, i, 6] * rows)
                heightFactor = frame.shape[0] / 300.0
                widthFactor = frame.shape[1] / 300.0
                x = int(widthFactor * xLeftBottom)
                y = int(heightFactor * yLeftBottom)
                h = int(widthFactor * xRightTop)
                w = int(heightFactor * yRightTop)
                cv2.rectangle(frame, (x, y), (h, w), (0, 255, 0))

                widht = w #int(w / 2) + x

                #if Distance_level < 10:
                #    Distance_level = 10
                if widht != 0:
                    Distance =5460.6 / widht #(5.7 * 958) #
                    print(int(Distance))
                    DistanceList.append(Distance)
                    avergDistnce = averageFinder(DistanceList, 6)
                    roundedDistance = round((avergDistnce*0.0254), 2)
                    # Drwaing Text on the screen
                    #Distance_level = int(Distance)

                    if jarakAwal != 0:
                        changeDistance = Distance - jarakAwal
                        distanceInMeters = changeDistance * 0.0254
                        velocity = distanceInMeters / changeInTime
                        speedList.append(velocity)
                        averageSpeed = averageFinder(speedList, 6)
                        print(averageSpeed)
                    jarakAwal = avergDistnce
                    changeInTime = time.time() - intialTime

                    if averageSpeed < 0:
                        averageSpeed = averageSpeed * -1
                        if averageSpeed >= 0 and averageSpeed <= 0.06:
                            cv2.putText(frame, "pelanggaran", (150, 150),  fonts, 2, (255,0,0),4)
    frame2 = cv2.resize(frame, (640,480))
    cv2.imshow("frame", frame2)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
