# Importing modules for gender_age_detection
from detection_methods.gender_and_age_detection import gender_age_detection
from detection_methods.gender_and_age_detection import opencv_keras_gender_age_detection
from detection_methods.image_detection import image_analysis

def main():
  input = input("Please enter what service you would like to use:")
  
  if input == "keras age and gender detection":
    opencv_keras_gender_age_detection.load_model()
  elif input == "normal age and gender detection":
    gender_age_detection.age_gender_detector()
  elif input == "image analysis example":
    image_analysis.Image()
  else: 
    print("Please restart the application, it is invalid")

if __name__ == "__main__":
  main()