import cv2
import numpy as np

# --- User: Set your scale here ---
pixels_per_cm = 37  # Example: 37 pixels = 1 cm (change this to your actual scale)

# Load image
img = cv2.imread("part2.png")
if img is None:
    print("Error: Image not found.")
    exit()

output = img.copy()

# --- Object Dimension Detection ---

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
length = h / pixels_per_cm
breadth = w / pixels_per_cm

# Draw bounding box and text for the main object
cv2.rectangle(output, (x, y), (x+ w, y + h), (0, 255, 0), 2)
cv2.putText(output, f"Length: {length:.2f} cm", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
cv2.putText(output, f"Breadth: {breadth:.2f} cm", (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

# --- Hole Diameter Detection ---
# Apply Gaussian blur for circle detection
blur = cv2.GaussianBlur(gray, (9, 9), 2)

# Detect circles using the Hough transform with tuned parameters
hole = cv2.HoughCircles(
    blur,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=100,
    param1=200,
    param2=20,
    minRadius=30,
    maxRadius=60
)

# Check if any circles were found
if hole is not None:
    hole = np.uint16(np.around(hole))

    for i in hole[0, :]:
        # Calculate the diameter in centimeters
        diameter_cm = (i[2] * 2) / pixels_per_cm

        # Draw the outer circle on the output image
        cv2.circle(output, (i[0], i[1]), i[2], (0, 255, 0), 2)

        # Draw the center of the circle
        cv2.circle(output, (i[0], i[1]), 2, (0, 0, 255), 3)

        # Display the diameter in cm, formatted to two decimal places
        cv2.putText(
            output,
            f"Diameter: {diameter_cm:.2f} cm",
            (i[0] - 50, i[1] + i[2] + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )
else:
    print("No circles found.")

# Show the final result
cv2.namedWindow("Detected Object and Holes", cv2.WINDOW_NORMAL)
cv2.imshow("Detected Object and Holes", output)
cv2.waitKey(0)
cv2.destroyAllWindows()