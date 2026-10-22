import cv2

# Load Haar Cascade classifier for cat face detection
cascade_path = 'haarcascade_frontalcatface_extended.xml'
cat_cascade = cv2.CascadeClassifier(cascade_path)

# Verify the cascade loaded correctly
if cat_cascade.empty():
    raise ValueError(f"Failed to load cascade classifier from {cascade_path}")
else:
    print(f"Successfully loaded Haar Cascade from {cascade_path}")
