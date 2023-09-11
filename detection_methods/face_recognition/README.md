# Face Recognition Python Script

This Python script utilizes the face_recognition library and OpenCV to perform face detection, calculate match percentages for known faces, and save unknown faces to a designated folder. The script uses a webcam as the video source to process frames in real-time.

## Requirements

- Python 3.x
- face_recognition
- OpenCV (cv2)
- numpy
- math
- datetime

## Installation

Ensure you have Python 3.x installed on your system. Install the required libraries using pip:

```
pip install face_recognition opencv-python numpy
```

## Usage

1. Create a folder named `faces` and place known face images (JPEG or PNG format) inside this folder. Each image should contain the face of a known person, and the filename will be used as the person's name during recognition.

2. Make sure your webcam is connected and accessible by the script.

3. Run the Python script using the following command:

```
python your_script_name.py
```

4. The script will open a window showing the webcam feed with face recognition annotations. It will also save the processed video frames with annotations and extract unknown faces to the `faces` folder.

5. Press 'q' on the keyboard to quit the script and close the video window.

Note: The script will process every other frame of video to save processing time.

## Function Explanation

### `face_confidence(face_distance, face_match_threshold=0.6)`

This function measures the confidence level (match percentage) of a face based on the distance between the face encoding of the detected face and the known faces. The default `face_match_threshold` is set to 0.6, meaning a face is considered a match if the distance is below this threshold. The function returns the match percentage as a string with two decimal places.

### `Face_Recognition` Class

This class handles the face recognition process.

#### `encode_faces()`

This method initializes the known face encodings by reading face images from the `faces` folder. It calculates the face encodings using the face_recognition library and stores them along with the corresponding names in `known_face_encodings` and `known_face_names`, respectively.

#### `save_faces(frame, face_locations)`

This method saves the unknown faces detected in each frame to the `faces` folder. It uses the face_recognition library to compare the detected face encoding with known face encodings and determine whether it matches any known face. If the face is unknown and not already saved, it will be saved with a filename in the format `UnknownN.jpg`, where N is a unique identifier.

#### `run_recognition()`

This method runs the face recognition process using the webcam feed. It continuously captures frames, processes every other frame for face recognition, and displays the results with annotations. The processed video frames are also saved in the `output_videos/face_recognition` folder with a filename based on the current date and time.

## Note

Ensure that you have the required folders (`faces` and `output_videos/face_recognition`) in the same directory as the script, and the webcam is accessible to the script. The face recognition process may be resource-intensive, so it is recommended to run the script on a device with a decent CPU/GPU.