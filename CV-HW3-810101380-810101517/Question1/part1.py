import cv2
import matplotlib.pyplot as plt

# Read input images
img1 = cv2.imread('cat1.webp')
img2 = cv2.imread('cat2.jpeg')

# Check if images loaded successfully
if img1 is None or img2 is None:
    raise ValueError("Failed to load one or both images")

# Display images
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
axes[0].set_title('Cat Image 1', fontsize=14)
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
axes[1].set_title('Cat Image 2', fontsize=14)
axes[1].axis('off')

plt.tight_layout()
plt.show()

print(f"Image 1 shape: {img1.shape}")
print(f"Image 2 shape: {img2.shape}")
