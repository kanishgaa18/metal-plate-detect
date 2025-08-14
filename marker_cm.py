import cv2
import numpy as np

# Define the dictionary and the known size of your Aruco marker
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
aruco_params = cv2.aruco.DetectorParameters()
marker_size = 5.1 

# Load image
img = cv2.imread("marked.png")
if img is None:
    print("Error: Image not found.")
    exit()
    
output = img.copy()

#Aruco Marker Detection for Calibration
corners, ids, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=aruco_params)

pixels_per = 0
if ids is not None:
    # Get the length of the first detected marker's side in pixels
    side_length_pixels = np.sqrt(np.sum((corners[0][0][0] - corners[0][0][1]) ** 2))
    pixels_per= side_length_pixels / marker_size
    print(f"Auto-calibrated: {pixels_per:.2f} pixels per cm")
    cv2.aruco.drawDetectedMarkers(output, corners, ids)
else:
    print("Error: Aruco marker not found. Cannot calibrate scale.")
    exit()

# Object Dimension Detection 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Find contours for the main object
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if not contours:
    print("No object found.")
    exit()

# Find the largest contour (the object's outer boundary)
c = max(contours, key=cv2.contourArea)

# Get bounding rectangle for the main object
x, y, w, h = cv2.boundingRect(c)

# Convert to centimeters
length = h / pixels_per
breadth = w / pixels_per

# Draw bounding box and text for the main object
cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.putText(output, f"Length: {length:.2f} cm", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
cv2.putText(output, f"Breadth: {breadth:.2f} cm", (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

# Apply Gaussian blur for circle detection
blur = cv2.GaussianBlur(gray, (9, 9), 2)

# Detect circles using the Hough transform with tuned parameters
hole = cv2.HoughCircles(blur,cv2.HOUGH_GRADIENT,dp=1,minDist=100,param1=200,param2=20,minRadius=26,maxRadius=28)

# Check if any circles were found
if hole is not None:
    hole = np.uint16(np.around(hole))

    for i in hole[0, :]:
        # Calculate the diameter in centimeters
        diameter = (i[2] * 2) / pixels_per

        # Draw the outer circle on the output image
        cv2.circle(output, (i[0], i[1]), i[2], (0, 255, 0), 2)

        # Draw the center of the circle
        cv2.circle(output, (i[0], i[1]), 2, (0, 0, 255), 3)

        # Display the diameter in cm, formatted to two decimal places
        cv2.putText(
            output,
            f"Diameter: {diameter:.2f} cm",
            (i[0] - 50, i[1] + i[2] + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )
else:
    print("No circles found.")

# --- 3. Quality Control Setup ---
# Define the true dimensions of the object in centimeters
true_length = 15.0
true_breadth = 15.0
true_hole_diameter = 1.20

# Define the tolerance in centimeters
tolerance = 0.5 

# Initially assume the quality check passes
quality = "PASS"
status = (0, 255, 0) # Green color for PASS

# Check the object's length
if not (true_length - tolerance <= length <= true_length + tolerance):
    quality = "FAIL"
    status = (0, 0, 255) 

# Check the object's breadth
if not (true_breadth - tolerance <= breadth <= true_breadth + tolerance):
    quality = "FAIL"
    status = (0, 0, 255) 

# Check the hole's diameter
if hole is not None:
    for i in hole[0, :]:
        if not (true_hole_diameter - tolerance <= diameter <= true_hole_diameter + tolerance):
            quality = "FAIL"
            status = (0, 0, 255) 

# Put the final quality status text 
cv2.putText(output,f"Quality: {quality}",(10, 30),cv2.FONT_HERSHEY_SIMPLEX,1,status, 2)

# Show the final result
cv2.namedWindow("Quality Check", cv2.WINDOW_NORMAL)
cv2.imshow("Quality Check", output)
cv2.waitKey(0)
cv2.destroyAllWindows()