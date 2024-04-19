import cv2
import numpy as np

# Specify some x amount of levels for pyramid-level synthesis
# At each pyramid level, identify a good 8 x 8 sample neighborhood. 

def generate_pyramid(image, levels=3):
    pyramid = [image]
    for _ in range(1, levels):
        image = cv2.pyrDown(image)
        pyramid.append(image)
    return pyramid

def process_block(block):
    return np.mean(block, axis=(0, 1))

def process_pyramid(pyramid):
    block_size = 8
    results = []
    for level in pyramid:
        level_results = []
        height, width, _ = level.shape
        for y in range(0, height, block_size):
            for x in range(0, width, block_size):
                block = level[y:y+block_size, x:x+block_size]
                if block.shape[0] == block_size and block.shape[1] == block_size:
                    result = process_block(block)
                    level_results.append(result)
        results.append(level_results)
    return results

def display_images(pyramid):
    for i, level in enumerate(pyramid):
        window_name = f'Pyramid Level {i}'
        cv2.imshow(window_name, level)
        cv2.waitKey(0)  # Wait for a key press to continue
        cv2.destroyWindow(window_name)

image = cv2.imread('textures/cobblestone.png')

pyramid = generate_pyramid(image)

# results = process_pyramid(pyramid)

display_images(pyramid)

cv2.destroyAllWindows()