import cv2 as cv

image = cv.imread('kitchen_restaurant.jpg')
image_hls = cv.cvtColor(image, cv.COLOR_BGR2HLS)
image_hue_180 = image_hls[:, :, 0]
image_hue = image_hue_180.copy()
image_hue *= 2

image_hue_normalized = image_hls.copy()
image_hue_normalized[:, :, 1] = 255 / 2
image_hue_normalized[:, :, 2] = 255

image_hue_normalized_bgr = cv.cvtColor(image_hue_normalized, cv.COLOR_HLS2BGR)
cv.imwrite('image_hue_normalized.png', image_hue_normalized_bgr)
