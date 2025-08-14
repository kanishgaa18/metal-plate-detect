import cv2
import numpy as np

def find_plate_dimensions(image_path, known_plate_width_cm):
    """
    Detects the dimensions of a rectangular metal plate and its circular holes,
    including spacing.

    This function reads an image, finds contours for the main plate and its holes,
    and then calculates their real-world dimensions and spacing based on a known
    reference dimension (the plate's own width) for calibration. The dimensions
    are returned in centimeters.

    Args:
        image_path (str): The path to the edge-detected image file.
        known_plate_width_cm (float): The known physical width of the plate
                                     in centimeters. This is used for calibration.

    Returns:
        tuple: A tuple containing the dimensions of the plate and a list of hole
               dimensions.
    """
    # 1. Image Preprocessing
    # Load the image as a single-channel grayscale image.
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Could not read image from path '{image_path}'.")
        return None, None
    
    # 2. Finding all contours
    # Use cv2.RETR_TREE to find all contours, including nested ones.
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("Error: No contours found in the image.")
        return None, None

    # 3. Find the main object contour and holes.
    # Combine all contours to get the bounding box of the entire object.
    all_points = np.concatenate(contours)
    
    # Get the bounding rectangle for the entire object
    x_main, y_main, w_main, h_main = cv2.boundingRect(all_points)

    # 4. Separate contours for the holes based on their area and position.
    hole_contours = []
    min_hole_area = 50  # Adjust as needed
    
    # Find contours that are inside the main bounding box and are not the main object itself
    for contour in contours:
        area = cv2.contourArea(contour)
        x_hole, y_hole, w_hole, h_hole = cv2.boundingRect(contour)
        
        # Filter by area and aspect ratio for circular holes
        if min_hole_area < area < (w_main * h_main) * 0.1 and abs(w_hole - h_hole) < 10:
             hole_contours.append(contour)
    
    # 5. Automatic Calibration
    if w_main <= 0 or h_main <= 0:
        print("Error: Detected plate dimensions are zero or less. Cannot calibrate.")
        return None, None
    
    pixel_to_cm_ratio = known_plate_width_cm / w_main

    # Convert the pixel dimensions to centimeters
    width_cm_main = w_main * pixel_to_cm_ratio
    height_cm_main = h_main * pixel_to_cm_ratio

    # 6. Calculate Dimensions for each hole
    hole_rects = []
    for contour in hole_contours:
        x, y, w, h = cv2.boundingRect(contour)
        hole_rects.append({'x': x, 'y': y, 'w': w, 'h': h})

    # Sort holes by x-coordinate to easily identify top-left, top-right, etc.
    hole_rects.sort(key=lambda r: r['x'])
    top_holes = sorted(hole_rects[:2], key=lambda r: r['y'])
    bottom_holes = sorted(hole_rects[2:], key=lambda r: r['y'])
    
    # 7. Draw all bounding boxes, lines, and text annotations
    display_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if display_image is None:
        print(f"Error: Could not read image from path '{image_path}' for display.")
        return None, None
    
    img_height, img_width = display_image.shape[:2]
    max_display_width = 1200
    max_display_height = 800
    scale = 1.0

    if img_width > max_display_width or img_height > max_display_height:
        scale = min(max_display_width / img_width, max_display_height / img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        display_image = cv2.resize(display_image, (new_width, new_height))
    
    # Scale coordinates for drawing
    x_main_scaled = int(x_main * scale)
    y_main_scaled = int(y_main * scale)
    w_main_scaled = int(w_main * scale)
    h_main_scaled = int(h_main * scale)

    # Draw the main object bounding box
    cv2.rectangle(display_image, (x_main_scaled, y_main_scaled), (x_main_scaled + w_main_scaled, y_main_scaled + h_main_scaled), (0, 255, 0), 2)
    
    # Draw hole bounding boxes and annotations
    for rect in hole_rects:
        x_hole_scaled = int(rect['x'] * scale)
        y_hole_scaled = int(rect['y'] * scale)
        w_hole_scaled = int(rect['w'] * scale)
        h_hole_scaled = int(rect['h'] * scale)
        cv2.rectangle(display_image, (x_hole_scaled, y_hole_scaled), (x_hole_scaled + w_hole_scaled, y_hole_scaled + h_hole_scaled), (0, 255, 0), 2)
        cv2.putText(display_image, f"{rect['w'] * pixel_to_cm_ratio:.2f} cm", (x_hole_scaled + 5, y_hole_scaled - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw main dimensions text
    cv2.putText(display_image, f"Width: {width_cm_main:.2f} cm", (x_main_scaled + w_main_scaled//2 - 100, y_main_scaled - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(display_image, f"Height: {height_cm_main:.2f} cm", (x_main_scaled + w_main_scaled + 20, y_main_scaled + h_main_scaled//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Calculate and draw spacing ONLY if 4 holes are found
    if len(hole_rects) == 4:
        # Top-left hole
        top_left_hole = top_holes[0]
        # Top-right hole
        top_right_hole = top_holes[1]
        # Bottom-left hole
        bottom_left_hole = bottom_holes[0]
        # Bottom-right hole
        bottom_right_hole = bottom_holes[1]

        # Horizontal spacing between holes
        horizontal_spacing_top_cm = ((top_right_hole['x'] - (top_left_hole['x'] + top_left_hole['w'])) * pixel_to_cm_ratio)
        horizontal_spacing_bottom_cm = ((bottom_right_hole['x'] - (bottom_left_hole['x'] + bottom_left_hole['w'])) * pixel_to_cm_ratio)

        # Vertical spacing between holes
        vertical_spacing_left_cm = ((bottom_left_hole['y'] - (top_left_hole['y'] + top_left_hole['h'])) * pixel_to_cm_ratio)
        vertical_spacing_right_cm = ((bottom_right_hole['y'] - (top_right_hole['y'] + top_right_hole['h'])) * pixel_to_cm_ratio)

        # Spacing from edges
        left_spacing_cm = (top_left_hole['x'] - x_main) * pixel_to_cm_ratio
        right_spacing_cm = ((x_main + w_main) - (top_right_hole['x'] + top_right_hole['w'])) * pixel_to_cm_ratio
        top_spacing_cm = (top_left_hole['y'] - y_main) * pixel_to_cm_ratio
        bottom_spacing_cm = ((y_main + h_main) - (bottom_left_hole['y'] + bottom_left_hole['h'])) * pixel_to_cm_ratio

        # Draw spacing lines and text (example for top-left hole)
        tl_x, tl_y, tl_w, tl_h = top_holes[0]['x'], top_holes[0]['y'], top_holes[0]['w'], top_holes[0]['h']
        tl_x_scaled, tl_y_scaled, tl_w_scaled, tl_h_scaled = int(tl_x * scale), int(tl_y * scale), int(tl_w * scale), int(tl_h * scale)

        # Top spacing
        cv2.line(display_image, (tl_x_scaled + tl_w_scaled//2, y_main_scaled), (tl_x_scaled + tl_w_scaled//2, tl_y_scaled), (0, 0, 255), 1)
        cv2.putText(display_image, f"{top_spacing_cm:.2f} cm", (tl_x_scaled + tl_w_scaled//2 + 5, y_main_scaled + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Left spacing
        cv2.line(display_image, (x_main_scaled, tl_y_scaled + tl_h_scaled//2), (tl_x_scaled, tl_y_scaled + tl_h_scaled//2), (0, 0, 255), 1)
        cv2.putText(display_image, f"{left_spacing_cm:.2f} cm", (x_main_scaled + 10, tl_y_scaled + tl_h_scaled//2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    else:
      print("Warning: Spacing calculations skipped as 4 holes were not detected.")

    cv2.imshow("Detected Metal Plate with Dimensions", display_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return (width_cm_main, height_cm_main), hole_rects

# --- Example Usage ---
image_file = 'edges_output.png'
known_plate_width = 50.0  # The real-world width of the plate in cm

plate_dims, hole_dims = find_plate_dimensions(image_file, known_plate_width)

if plate_dims is not None and hole_dims is not None:
    print("\n--- Results ---")
    print(f"Plate Dimensions: Width = {plate_dims[0]:.2f} cm, Height = {plate_dims[1]:.2f} cm")
    print("Hole Dimensions:")
    for i, dims in enumerate(hole_dims):
        print(f"  Hole {i+1}: Width = {dims['w'] * (known_plate_width/plate_dims[0]):.2f} cm, Height = {dims['h'] * (known_plate_width/plate_dims[0]):.2f} cm")
