import cv2
import numpy as np
from matplotlib import pyplot as plt

def non_max_suppression(boxes, overlap_thresh=0.3):
    if len(boxes) == 0:
        return []
    
    boxes = np.array(boxes)
    pick = []
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    area = boxes[:, 2] * boxes[:, 3]
    
    idxs = np.argsort(area)[::-1]
    
    while len(idxs) > 0:
        i = idxs[0]
        pick.append(i)
        
        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        
        overlap = (w * h) / area[idxs[1:]]
        
        idxs = np.delete(idxs, np.concatenate(([0], np.where(overlap > overlap_thresh)[0] + 1)))
    
    return boxes[pick].astype(int)

def detect_cat_faces(image, apply_nms=True):
    cat_cascade = cv2.CascadeClassifier('haarcascade_frontalcatface_extended.xml')
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.equalizeHist(gray)
    
    faces = cat_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )
    
    if apply_nms and len(faces) > 1:
        faces = non_max_suppression(faces, overlap_thresh=0.4)
    
    result = image.copy()
    
    for (x, y, w, h) in faces:
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(result, 'Cat', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, (0, 255, 0), 2)
    
    return result, len(faces)

def rotate_image(image, angle):
    """Rotate image by specified angle (90, 180, 270 degrees)"""
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return image

# Load images
img1 = cv2.imread('cat1.webp')
img2 = cv2.imread('cat2.jpeg')

# Original images
img1_original, count1_original = detect_cat_faces(img1, apply_nms=True)
img2_original, count2_original = detect_cat_faces(img2, apply_nms=True)

# Rotate 90 degrees
img1_rotated = rotate_image(img1, 90)
img2_rotated = rotate_image(img2, 90)

img1_rotated_result, count1_rotated = detect_cat_faces(img1_rotated, apply_nms=True)
img2_rotated_result, count2_rotated = detect_cat_faces(img2_rotated, apply_nms=True)

# Display results
plt.figure(figsize=(16, 12))

plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(img1_original, cv2.COLOR_BGR2RGB))
plt.title(f'Image 1 - Original (Detected: {count1_original})')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(img1_rotated_result, cv2.COLOR_BGR2RGB))
plt.title(f'Image 1 - Rotated 90° (Detected: {count1_rotated})')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(img2_original, cv2.COLOR_BGR2RGB))
plt.title(f'Image 2 - Original (Detected: {count2_original})')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(img2_rotated_result, cv2.COLOR_BGR2RGB))
plt.title(f'Image 2 - Rotated 90° (Detected: {count2_rotated})')
plt.axis('off')

plt.tight_layout()
plt.show()

print("=" * 60)
print("Rotation Resistance Analysis:")
print("=" * 60)
print(f"\nImage 1:")
print(f"  - Original: {count1_original} face(s) detected")
print(f"  - Rotated 90°: {count1_rotated} face(s) detected")

print(f"  - Original: {count2_original} face(s) detected")
print(f"  - Rotated 90°: {count2_rotated} face(s) detected")



