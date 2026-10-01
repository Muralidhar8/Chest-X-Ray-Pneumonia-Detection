"""Creates placeholder asset for frontend initialization"""
import os
import cv2
import numpy as np

os.makedirs("app/assets", exist_ok=True)
img = np.zeros((400, 400), dtype=np.uint8)
# Subtle thoracic contour
cv2.ellipse(img, (200, 200), (120, 160), 0, 0, 360, 40, -1)
cv2.imwrite("app/assets/placeholder_xray.png", img)
print("Placeholder saved.")
