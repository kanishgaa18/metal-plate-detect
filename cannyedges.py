import cv2

image = cv2.imread('metal1.png')  # Replace with your image path


gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray_image, 200,500)

#cv2.namedWindow('Canny Edges', cv2.WINDOW_NORMAL)

#cv2.imshow('Original Image', image)
cv2.imshow('Canny Edges', edges)

#cv2.imwrite('edges_output4.png', edges)

cv2.waitKey(0)
cv2.destroyAllWindows()
