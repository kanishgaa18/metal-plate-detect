import cv2
import numpy as np

# Choose the dictionary and marker ID
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
marker_id = 24 # You can choose any ID from the dictionary
marker_size_pixels = 200 # Adjust this for a larger or smaller image

# Generate the marker image
marker_image = np.zeros((marker_size_pixels, marker_size_pixels), dtype=np.uint8)
cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size_pixels, marker_image, 1)

# Save the marker image
cv2.imwrite(f"aruco_marker.png", marker_image)
#print(f"Marker image saved as aruco_marker_id_{marker_id}.png")

cv2.imshow("Aruco Marker", marker_image)
cv2.waitKey(0)
cv2.destroyAllWindows()