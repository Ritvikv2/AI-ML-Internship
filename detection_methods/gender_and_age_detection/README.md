## Gender and Age Detection with Logo Insertion

This repository contains code to perform real-time gender and age detection on faces captured by a camera. The detected faces will have bounding boxes drawn around them, with the predicted gender and age labels displayed above each face. Additionally, a logo will be inserted in the bottom right corner of the frame. The processed frames are also saved into a video file.

### Requirements
- Python 3
- OpenCV (cv2) library
- Numpy library

### Instructions

1. Clone the repository to your local machine or download the script and the required files.

2. Install the necessary libraries (if not already installed) using pip:
   ```
   pip install opencv-python
   pip install numpy
   ```

3. Download the YOLOv7 configuration and weights files from the official YOLO website and place them in the repository directory:
   - yolov7.cfg
   - yolov7.weights

4. Download the face detection, age detection, and gender detection model files and place them in the corresponding directories:
   - age_gender_detector/opencv_face_detector.pbtxt
   - age_gender_detector/opencv_face_detector_uint8.pb
   - age_gender_detector/age_deploy.prototxt
   - age_gender_detector/age_net.caffemodel
   - age_gender_detector/gender_deploy.prototxt
   - age_gender_detector/gender_net.caffemodel

5. Add age group images to the 'detection_methods/gender_and_age_detection/Advertisements/' directory. The images should be named according to the age groups in the ageList variable.

6. Add the logo image to the 'detection_methods/gender_and_age_detection/Advertisements/' directory with the filename 'logo.jpg'.

7. Adjust the 'padding' value as needed to fine-tune the face detection bounding boxes.

8. Run the script:
   ```
   python gender_age_detection.py
   ```

9. The camera feed will open, and you should see faces with bounding boxes and gender and age labels. The processed frames are also saved in a video file in the 'output_videos/gender_and_age_detection/' directory with a filename corresponding to the date and time.

10. Press 'q' to stop the script and exit the camera feed window.

### Note
- The accuracy of gender and age predictions may vary based on the quality of the input camera feed and the pre-trained models used.

- The script uses the YOLOv7 object detection model for face detection and separate pre-trained models for gender and age detection. Make sure to download the correct model files.

- Ensure that all the required files and directories are present in the correct locations before running the script.

- For best results, use a camera with a good resolution and adequate lighting conditions.