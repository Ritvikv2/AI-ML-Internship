'''
This python file includes, detecting faces, match percentage and also saves unkown faces into the faces file. 
'''

# Importing modules
import face_recognition
import os, sys
import cv2
import numpy as np
import math
import datetime

# Generalised Function - ensures face is within the faces value
def face_confidence(face_distance, face_match_threshold=0.6): # default value of match 
    range_val = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range_val * 2.0) # normalised distance faces (linear value 0-1) where 1 is P match

    # Check if the face distance exceeds the threshold
    if face_distance > face_match_threshold:
        # If above the threshold, return the linear value as a percentage with 2 decimal places
        return str(round(linear_val * 100, 2)) + '%'
    else:
        # If below or equal to the threshold, apply a non-linear transformation to the linear value
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        # Return the transformed value as a percentage with 2 decimal places
        return str(round(value, 2)) + '%'

# Class for Face_Recognition
class Face_Recognition:
    # General lists used in the Face_Recognition class
    face_locations = []  # Stores the locations of detected faces
    face_encodings = []  # Stores the encodings of detected faces
    face_names = []  # Stores the names of detected faces
    known_face_encodings = []  # Stores the encodings of known faces
    known_face_names = []  # Stores the names of known faces
    saved_face_names = set()  # Set to keep track of saved face names
    process_current_frame = True  # Flag to indicate whether to process the current frame or not
    output_directory = 'detection_methods/face_recognition/faces'  # Directory to save extracted faces

    # Attributes with particular instances
    def __init__(self):
        self.encode_faces()  # Initialize the known face encodings

    # This function is used to measure the similarity between the two face images
    def encode_faces(self):
        image_directory = 'detection_methods/face_recognition/faces'  # Path to the directory containing face images
        valid_extensions = ('.jpg', '.jpeg', '.png')  # Valid file extensions for face images

        for filename in os.listdir(image_directory):
            if filename.endswith(valid_extensions):
                image_path = os.path.join(image_directory, filename)  # Get the full path to the image file
                face_image = face_recognition.load_image_file(image_path)  # Load the image file using face_recognition library

                face_encodings = face_recognition.face_encodings(face_image)

                if len(face_encodings) > 0:
                    face_encoding = face_encodings[0]
                else:
                    # Handle the case when no face is found in the image
                    print("No face found in the image.")


                face_encoding = face_recognition.face_encodings(face_image)[0]  # Encode the face in the image - similarity between image

                self.known_face_encodings.append(face_encoding)  # Append the face encoding to the list of known face encodings
                self.known_face_names.append(filename)  # Append the filename (name of the person) to the list of known face names

        print(self.known_face_names)  # Print the names of the known faces (present in the faces file)

    # This function saves the faces captured in the frame
    def save_faces(self, frame, face_locations):
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)  # Create the output directory if it doesn't exist

        for (top, right, bottom, left) in face_locations:
            # Scale back up face locations to the original frame size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Extract the face image from the frame
            face_image = frame[top:bottom, left:right]

            # Check if the face is already saved
            face_encodings = face_recognition.face_encodings(face_image)
            if len(face_encodings) == 0:
                print("No face found in the extracted image. Skipping...")
                continue

            face_encoding = face_encodings[0]  # Encode the face in the extracted image
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)  # Compare the face encoding with known face encodings
            name = "Unknown"
            confidence = '???'
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)  # Calculate the distance between the face and known faces
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
                confidence = face_confidence(face_distances[best_match_index])  # Get the confidence level of the best match face

            # Save the face only if it has not been saved before
            if name not in self.known_face_names:
                
                if name in self.saved_face_names:
                    name += str(len(self.saved_face_names))  # check - Unknown1, Unknown2, ...

                self.saved_face_names.add(name)
                filename = f'{self.output_directory}/{name}.jpg'  # Generate the output filename based on the face name
                cv2.imwrite(filename, face_image)  # Save the face image as a file

    # Function that runs the recognition using all the functions combined
    def run_recognition(self):
        video_capture = cv2.VideoCapture(1)  # Open the video capture (webcam)

        # Retrieve the width and height of the video frames - save video
        width= int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height= int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create a VideoWriter object to save the processed video frames (datetime)
        writer = cv2.VideoWriter(f'output_videos/face_recognition/{datetime.datetime.now()}.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20, (width,height))

        if not video_capture.isOpened():
            sys.exit('Video source not found...')  # Exit the program if the video source is not found
        
        while True:
            ret, frame = video_capture.read()  # Read a frame from the video capture (webcam)

            # Only process every other frame of video to save time
            if self.process_current_frame:
                # Resize frame of video to 1/4 size for faster face recognition processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

                # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                rgb_small_frame = small_frame[:, :, ::-1]

                # Find all the faces and face encodings in the current frame of video
                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

                self.face_names = []
                for face_encoding in self.face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = "Unknown"
                    confidence = '???'

                    # Calculate the shortest distance to a known face
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = face_confidence(face_distances[best_match_index])

                    self.face_names.append(f'{name} ({confidence})')


                # Save the extracted faces
                self.save_faces(frame, self.face_locations)

            self.process_current_frame = not self.process_current_frame

            # Display the results
            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                # Create a rectangle around the face and display the name and confidence
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame , name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

            # Write the frame to the output video file
            writer.write(frame)

            # Display the resulting image
            cv2.imshow('Face Recognition 1', frame)


            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) == ord('q'):
                break

        # Release the webcam and destroy any remaining windows
        video_capture.release()
        cv2.destroyAllWindows()