from PIL import Image

def find_lefts_and_rights(image):
    ''' Determines the startx and endx of all digits in a line '''
    (w, h) = image.size
    digit_locations = [] # Each element contains the startx and endx of a digit
    new_digit_loc = [] # The startx and endx of the digit the algorithm is currently working on
    searching_for_beginning = True
    wsc = 0 # Number of columns of white space the algorithm has passed through after the last instance of black in a digit
    for x in range(w):
        for y in range(h):
            if searching_for_beginning:                  # If we're looking for a new digit,
                if image.getpixel((x, y)) == 0:              # and we find black,
                    new_digit_loc.append(x)                  # that means we've found the start of a new digit, so add that xpos to a list,
                    searching_for_beginning = False          # and stop searching for the start of new digits for now.
            else:                                        # If we're looking for the end of the current digit,
                if image.getpixel((x, y)) == 0:              # and we find black, we're not done with the current digit,
                    break                                    # so move on to the next column.
        if not searching_for_beginning:                  # If we find nothing of interest while reading a digit,
            wsc += 1                                         # that means we've hit a line of whitespace,
            if wsc > limit:                                  # and if we've hit (limit) lines of whitespace already,
                new_digit_loc.append(x - wsc)                # then we're done reading the current digit, so set the end xpos of the digit to the last column that had black in it,
                digit_locations.append(new_digit_loc)        # add the startx & endx list to the list of digit locations,
                wsc = 0                                      # reset the whitespace counter,
                searching_for_beginning = True               # and start searching for new digits.

    return digit_locations

def find_top_and_bottom(image, startx, endx):
    ''' Determines the starty and endy of a single digit, given its startx and endx '''
    h = image.size[1]
    searching_for_beginning = True
    wsc = 0
    for y in range(h):
        for x in range(startx, endx + 1):
            if searching_for_beginning:
                if image.getpixel((x, y)) == 0:
                    starty = y
                    searching_for_beginning = False
            else:
                if image.getpixel((x, y)) == 0:
                    break
        if not searching_for_beginning:
            wsc += 1
            if wsc > limit:
                endy = y - wsc
                break
    return [starty, endy]

def process_input_image(image):
    ''' Takes an image of a line of characters and returns a list of images of each individual character '''
    digit_xpositions = find_lefts_and_rights(image)
    digit_ypositions = []
    for item in digit_xpositions:
        digit_ypositions.append(find_top_and_bottom(image, item[0], item[1]))

    digit_images = []
    for i in range(digit_xpositions):
        digit_images.append(image.crop((digit_xpositions[i][0], digit_ypositions[i][0], digit_xpositions[i][1], digit_ypositions[i][1])))
