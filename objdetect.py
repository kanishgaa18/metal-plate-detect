import cv2
import numpy as np

# --- User: Set your scale here ---
pixels_per_cm = 40  # Example: 37 pixels = 1 cm (change this to your actual scale)

# Load image
img = cv2.imread("metal1.png")  # Change to your image filename
if img is None:
    print("Error: Image not found.")
    exit()

# Convert to grayscale and threshold
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if not contours:
    print("No object found.")
    exit()

# Find the largest contour 
c = max(contours, key=cv2.contourArea)

# Get bounding rectangle
x, y, w, h = cv2.boundingRect(c)

# Convert to centimeters
length_cm = h / pixels_per_cm
breadth_cm = w / pixels_per_cm

# Draw bounding box
cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Put text for length and breadth in cm
cv2.putText(img, f"Length: {length_cm:.2f} cm", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
cv2.putText(img, f"Breadth: {breadth_cm:.2f} cm", (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

# Apply median blur to reduce noise
blur = cv2.medianBlur(gray, 5)

# Detect circles using the Hough transform
hole = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 50, param1=250, param2=25, minRadius=0, maxRadius=0)

# Check if any circles were found
if hole is not None:
    hole = np.uint16(np.around(hole))

    for i in hole[0, :]:
        # Calculate the radius in centimeters
        radius_cm = (i[2] * 2 )/ pixels_per_cm

        # Draw the outer circle on the original color image
        cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)

        # Draw the center of the circle
        cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)

        # Display the radius in cm, formatted to two decimal places
        cv2.putText(img, f"Radius: {radius_cm:.2f} cm", (i[0] - 50, i[1] + i[2] + 20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255, 0, 0),1)
else:
    print("No circles found.")

# Show result in a resizable window
#cv2.namedWindow("Edge Detected", cv2.WINDOW_NORMAL)
cv2.imshow("Edge Detected", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
