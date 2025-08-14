'''import cv2
import numpy as np

def find_plate_dimensions(image_path, known_plate_width_cm):

    # 1. Image Preprocessing
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from path '{image_path}'.")
        return None, None

    # Get image dimensions
    img_height, img_width = image.shape[:2]
    max_display_width = 1200
    max_display_height = 800

    # Resize the image if it's too large for the display
    if img_width > max_display_width or img_height > max_display_height:
        scale = min(max_display_width / img_width, max_display_height / img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        resized_image = cv2.resize(image, (new_width, new_height))
    else:
        resized_image = image.copy()

    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 2. Finding the main metal plate contour
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    largest_contour = None
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            largest_contour = contour

    if largest_contour is None:
        print("Error: No contours found for the plate.")
        return None, None

    # 3. Calculating Dimensions from the original contour
    rect = cv2.minAreaRect(largest_contour)
    
    width_pixels = rect[1][0]
    height_pixels = rect[1][1]

    # 4. Automatic Calibration
    # The pixel-to-cm ratio is calculated based on the assumption that
    # the detected plate's width matches the 'known_plate_width_cm'.
    if width_pixels <= 0:
        print("Error: Detected plate width is zero or less. Cannot calibrate.")
        return None, None
        
    pixel_to_cm_ratio = known_plate_width_cm / width_pixels

    # Convert the pixel dimensions to centimeters
    width_cm = width_pixels * pixel_to_cm_ratio
    height_cm = height_pixels * pixel_to_cm_ratio

    # 5. Draw the bounding box and text on the image
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    cv2.drawContours(resized_image, [box], 0, (0, 255, 0), 2)
    
    cv2.putText(resized_image, f"Width: {width_cm:.2f} cm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(resized_image, f"Height: {height_cm:.2f} cm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow("Detected Metal Plate with Dimensions", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return width_cm, height_cm

# --- Example Usage ---
# Now you only need to provide the known physical width of the plate.
# The code will automatically find the corresponding pixel dimension for calibration.
image_file = 'metal.png'  # <-- Replace with your image path
known_plate_width = 50.0  # The real-world width of the plate in cm

width, height = find_plate_dimensions(image_file, known_plate_width)

if width is not None and height is not None:
    print("\n--- Results ---")
    print(f"Plate Dimensions: Width = {width:.2f} cm, Height = {height:.2f} cm")
'''