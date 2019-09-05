import numpy as np
import PIL.ImageOps
from skimage.transform import resize

limit = 5 

def find_lefts_and_rights(image):
    """ Determines the startx and endx of all digits in a line """
    (w, h) = image.size
    digit_locations = [] # Each element contains the startx and endx of a digit
    new_digit_loc = [] # The startx and endx of the digit the algorithm is currently working on
    searching_for_beginning = True
    wsc = 0 # Number of columns of white space the algorithm has passed through after the last instance of black in a digit
    for x in range(w):
        nothing_found = True
        for y in range(h):
            if searching_for_beginning:                  # If we're looking for a new digit,
                if image.getpixel((x, y)) == 255:            # and we find white,
                    new_digit_loc.append(x)                  # that means we've found the start of a new digit, so add that xpos to a list,
                    searching_for_beginning = False          # and stop searching for the start of new digits for now.
            else:                                        # If we're looking for the end of the current digit,
                if image.getpixel((x, y)) == 255:            # and we find black, we're not done with the current digit,
                    nothing_found = False                    # and obviously we've found something,
                    break                                    # so move on to the next column without incrementing wsc.
        if not searching_for_beginning and nothing_found: # If we find nothing of interest while reading a digit,
            wsc += 1                                         # that means we've hit a line of blackspace,
            if wsc > limit:                                  # and if we've hit (limit) lines of blackspace already,
                new_digit_loc.append(x - wsc)                # then we're done reading the current digit, so set the end xpos of the character to the last instance of white
                digit_locations.append(new_digit_loc)        # add the startx & endx list to the list of digit locations,
                wsc = 0                                      # reset the blackspace counter,
                new_digit_loc = []                           # reset the digit locations list,
                searching_for_beginning = True               # and start searching for new digits.

    return digit_locations


def find_top_and_bottom(image, startx, endx):
    """ Determines the starty and endy of a single digit, given its startx and endx """

    h = image.size[1]
    searching_for_beginning = True
    wsc = 0
    starty = None
    endy = None
    for y in range(h):
        nothing_found = True
        for x in range(startx, endx + 1):
            if searching_for_beginning:
                if image.getpixel((x, y)) == 255:
                    starty = y
                    nothing_found = False
                    searching_for_beginning = False
            else:
                if image.getpixel((x, y)) == 255:
                    nothing_found = False
                    break
        if not searching_for_beginning and nothing_found:
            wsc += 1
            if wsc > limit:
                endy = y - wsc
                break
    if starty != endy:
        return [starty, endy]
    print("A dot was found at (" + str(startx) + ", " + str(starty) + "); your image needs cleaning!!!")
    input("Press enter if you wish to continue, but the program WILL crash!!! ")
    return [0, image.size[1] - 1]


def format_image(image_ndarray):
    """ Formats an image for input into the model """

    # Resize the image to have the same resolution as the training data
    image_ndarray = resize(image_ndarray, (20, 20), anti_aliasing=True)

    # Add four pixels of padding to all sides of the image
    image_list_temp = [[0 for x in range(28)] for y in range(28)]
    for x in range(image_ndarray.shape[0]):
        for y in range(image_ndarray.shape[1]):
            image_list_temp[4 + y][4 + x] = image_ndarray[y][x]
    image_ndarray = np.array(image_list_temp)

    return image_ndarray


def convert_to_bw(image, threshold):
    """ Converts an image to black and white given a threshold """
    image = image.convert('L')
    for x in range(image.size[0]):
        for y in range(image.size[1]):
            if image.getpixel((x, y)) < threshold:
                image.putpixel((x, y), 0)
            else:
                image.putpixel((x, y), 255)
    image = PIL.ImageOps.invert(image)
    image.show()
    return image


def split_input_image(image):
    """ Takes an image of a line of characters and returns a list of images of each individual character """
    image = convert_to_bw(image, 80)
    digit_xpositions = find_lefts_and_rights(image) # WHY ARE YOU RETURNING NOTHING
    digit_ypositions = []
    for item in digit_xpositions:
        digit_ypositions.append(find_top_and_bottom(image, item[0], item[1]))

    digit_images = []
    for i in range(len(digit_xpositions)):
        digit_images.append(np.array(image.crop((digit_xpositions[i][0], digit_ypositions[i][0], digit_xpositions[i][1], digit_ypositions[i][1]))))

    return np.array(digit_images)
