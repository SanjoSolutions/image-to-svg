import cv2 as cv
import numpy as np

image = cv.imread('kitchen_restaurant_small.png')
image_hls = cv.cvtColor(image, cv.COLOR_BGR2HLS)
image_hue_180 = image_hls[:, :, 0]
image_hue = image_hue_180.copy()
image_hue *= 2

image_hue_normalized = image_hls.copy()
image_hue_normalized[:, :, 1] = 255 / 2
image_hue_normalized[:, :, 2] = 255

image_hue_normalized_bgr = cv.cvtColor(image_hue_normalized, cv.COLOR_HLS2BGR)
cv.imwrite('image_hue_normalized.png', image_hue_normalized_bgr)


class Region:
    def __init__(self, bucket, shape):
        self.bucket = bucket
        self.colors = set()
        self.mask = np.zeros(shape, dtype=np.bool8)


def find_regions(image):
    regions = []
    is_pixel_in_a_region = np.zeros(image.shape[:2], dtype=np.bool8)

    def is_pixel_not_in_a_region(position):
        return not is_pixel_in_a_region[position]

    def filter_positions_not_in_a_region(positions):
        return set(filter(is_pixel_not_in_a_region, positions))

    width = image.shape[1]
    height = image.shape[0]
    for position in generate_positions(width, height):
        if is_pixel_not_in_a_region(position):
            color = image[position]
            bucket = determine_color_bucket(color)
            region = Region(bucket, image.shape)
            region.colors.add(color)
            region.mask[position] = True
            is_pixel_in_a_region[position] = True

            next_neighbouring_positions = filter_positions_not_in_a_region(
                determine_neighbouring_positions(position, width, height)
            )
            while len(next_neighbouring_positions) >= 1:
                neighbouring_positions = next_neighbouring_positions
                next_neighbouring_positions = set()

                for position in neighbouring_positions:
                    color = image[position]
                    bucket = determine_color_bucket(color)
                    if bucket == region.bucket:
                        region.colors.add(color)
                        region.mask[position] = True
                        is_pixel_in_a_region[position] = True
                        next_neighbouring_positions |= determine_neighbouring_positions(position, width, height)

                next_neighbouring_positions = filter_positions_not_in_a_region(next_neighbouring_positions)

            regions.append(region)

    return regions


def generate_positions(width, height):
    for row in range(height):
        for column in range(width):
            position = (row, column)
            yield position


def determine_color_bucket(hue):
    if hue >= 329 or hue <= 14:
        bucket = 0
    elif hue <= 53:
        bucket = 1
    elif hue <= 67:
        bucket = 2
    elif hue <= 157:
        bucket = 3
    elif hue <= 191:
        bucket = 4
    elif hue <= 270:
        bucket = 5
    elif hue <= 295:
        bucket = 6
    elif hue <= 318:
        bucket = 7
    return bucket


def determine_neighbouring_positions(position, width, height):
    positions = set()

    row, column = position

    if row >= 1:
        positions.add((row - 1, column))
    if column <= width - 2:
        positions.add((row, column + 1))
    if row <= height - 2:
        positions.add((row + 1, column))
    if column >= 1:
        positions.add((row, column - 1))

    return positions


regions = find_regions(image_hue)


def visualize_regions(regions, width, height):
    visualization = np.zeros((height, width, 3), dtype=np.uint8)
    for region in regions:
        color = int(average(region.colors))
        visualization[region.mask] = (color, 255 / 2, 255)
    return visualization


def average(values):
    return float(sum(values)) / len(values)


visualization = visualize_regions(regions, image_hue.shape[1], image_hue.shape[0])
visualization_output = cv.cvtColor(visualization, cv.COLOR_HLS2BGR)
cv.imwrite('visualization.png', visualization_output)
