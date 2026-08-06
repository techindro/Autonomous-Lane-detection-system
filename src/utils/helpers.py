import cv2
import numpy as np

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def gaussian_blur(img, kernel=(5, 5)):
    return cv2.GaussianBlur(img, kernel, 0)

def canny_edge(img, low=50, high=150):
    gray = grayscale(img)
    blur = gaussian_blur(gray)
    return cv2.Canny(blur, low, high)

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(img, mask)

def draw_lines(img, lines, color=[255, 0, 0], thickness=12):
    line_img = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            if line is not None and isinstance(line, (np.ndarray, list, tuple)) and len(line) == 4:
                x1, y1, x2, y2 = np.array(line).reshape(4)
                cv2.line(line_img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    return line_img
