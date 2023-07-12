'''
This code has taken some inspiration and bases from (spe)
https://circuitdigest.com/microcontroller-projects/age-and-gender-classification-using-opencv-and-deep-learning-on-raspberry-pi

'''

# This is the file that functions properly to detect one's age
import cv2 as cv
import time

# Function to get face bounding boxes using a pre-trained face detection model
def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    # Create a blob (Binary Large object) from the input frame for the face detection model
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    # Perform forward pass to get detections
    detections = net.forward()
    bboxes = []
    # Iterate over the detections and filter out weak ones based on confidence threshold
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            # Get the coordinates of the bounding box
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            # Draw the bounding box on the frame
            cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes

# Paths for YOLOv7 configuration and weights
yoloConfigPath = "yolov7.cfg"
yoloWeightsPath = "yolov7.weights"
yoloNet = cv.dnn.readNetFromDarknet(yoloConfigPath, yoloWeightsPath)

# Paths for face detection, age detection, and gender detection models      
faceProto = "age_gender_detector/opencv_face_detector.pbtxt"
faceModel = "age_gender_detector/opencv_face_detector_uint8.pb"
ageProto = "age_gender_detector/age_deploy.prototxt"
ageModel = "age_gender_detector/age_net.caffemodel"
genderProto = "age_gender_detector/gender_deploy.prototxt"
genderModel = "age_gender_detector/gender_net.caffemodel"

# List of age buckets and genders
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746) # pre-processing the face image before feeding it into a model
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Load the age and gender detection models
ageNet = cv.dnn.readNet(ageModel, ageProto)
genderNet = cv.dnn.readNet(genderModel, genderProto)
faceNet = cv.dnn.readNet(faceModel, faceProto)

# Function takes a frame as an input and performs age and gender detection on faces in frame
def age_gender_detector(frame):
    # Read frame
    t = time.time()
    frameFace, bboxes = getFaceBox(faceNet, frame)
    for bbox in bboxes:
        # Extract the face region from the frame
        face = frame[max(0, bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
 
        # Preprocess the face for gender detection
        blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]

        # Preprocess the face for age detection
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]

        label = "{},{}".format(gender, age)
        cv.putText(frameFace, label, (bbox[0], bbox[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)
    return frameFace

padding = 20

# Open video capture
video = cv.VideoCapture(1)
# video = cv.VideoCapture('videos/NegativeCases/B001.mp4')
# video = cv.VideoCapture('videos/b.mp4')

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Perform face detection and age/gender prediction
    frame, bboxs = getFaceBox(faceNet, frame)

    for bbox in bboxs:
        # Extract the face region from the frame
        face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
                     max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]
        # Preprocess the face for gender detection
        blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPred = genderNet.forward()
        gender = genderList[genderPred[0].argmax()]

        # Preprocess the face for age detection
        ageNet.setInput(blob)
        agePred = ageNet.forward()
        age = ageList[agePred[0].argmax()]

        label = "{},{}".format(gender, age)
        cv.rectangle(frame, (bbox[0], bbox[1] - 30), (bbox[2], bbox[1]), (0, 255, 0), -1)
        cv.putText(frame, label, (bbox[0], bbox[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv.LINE_AA)

    cv.imshow("Age-Gender", frame)
    if cv.waitKey(1) == ord('q'):
        break

# Release video capture and destroy windows
video.release()
cv.destroyAllWindows()