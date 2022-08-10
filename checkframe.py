import numpy as np
import matplotlib.pyplot as plt

# Draw a rectangle given the image, center, extents
# My favourite mint color :))))
COLOR = (62, 180, 137)

# Plot the image
def plot(img):
    plt.figure()
    plt.imshow(img)
    plt.show()

# Show the corresponding coordinates on the image
# Coords is a (N, 2) numpy array, img is image
def check_frame(coords, img, dot_size = 3, base_brightness = 0.5):
    # Make sure the image is in uint8
    assert img.dtype == np.uint8, "The image must be in unsigned integer format"

    # Make a mask
    im = np.zeros(img.shape[:-1], dtype = float) + base_brightness

    # Loop through every coordinates, mark the coordinate (and its neighbouring coordinates) 1 on the mask
    for i in range(coords.shape[0]):
        c = coords[i]
        x = int(c[1])
        y = int(c[0])

        im[x, y] = 1
        for m in range(x-dot_size//2, x+dot_size//2+1):
            for n in range(y-dot_size//2, y+dot_size//2+1):
                try:
                    im[m, n] = 1
                except:
                    continue
    
    # Make a copy of the image to avoid changing the original via referece
    image = np.array(img, dtype = float)

    for i in range(3):
        image[:, :, i] *= im
        image[:, :, i][im == 1] = 255
    
    # Make it back into integer
    image = np.array(image, dtype = np.uint8)
    
    # Plot he image
    plot(image)

# image is the image, bbs is the bounding boxes, in a numpy array of shape (N, 4)
# bbs in the order left, top, right, bottom
def check_bounding_box(img, bbs, base_brightness = 0.5):
    
    new_img = np.array(img, dtype = float)
    new_img *= base_brightness
    new_img = np.array(new_img, dtype = img.dtype)

    col = np.array(COLOR, dtype = np.uint8)

    for i in range(bbs.shape[0]):
        lef, top, rig, bot = bbs[i]

        new_img[top, lef : rig, :] = col
        new_img[bot, lef : rig, :] = col
        new_img[top : bot, lef, :] = col
        new_img[top : bot, rig, :] = col
    
    plot(new_img)