# This processes the unity dataset data.txt. Upon feeding the original string into this function it will return a dictionary of
# N keys, where N is the number of frames in the dataset (should be 300?). The frame number would be the keys and the values are tuple in the
# format: (K = number of people, frame rate ( = 1/time between last frame and this frame ), frame data)

# and then frame data is another list of length K, and each element is a tuple in the format:
# (x coord, y coord, id to keep track which people is which, bounding box (bb) left, bb top, bb right, bb bottom)

# Feel free to copy and paste this :) unless you want to do string manipulation yourself :)))))

def process_unity_data(data: str):
    # Split all the data into frames first. The "frameend" is here exactly for this purpose. Discard anything after the last frameend, typically white space
    frs = data.split("frameend")[:-1]
    # Initialize empty dict to populate
    data_dict = {}

    # Loop through every frame
    for fr in frs:
        # Declare necessary variables
        frame_idx, num_people, frame_rate = 0, 0, 0
        frame_data_lis = []

        # Split rows first
        frdata = fr.split("\n")
        for f in frdata:
            # Skip the empty lines
            if len(f.strip()) == 0:
                continue

            # If f begins with "Frame " then extract the frame number
            if f[:12] == "Frame rate: ":
                frame_rate = float(f.split("Frame rate: ")[1])
                continue

            if f[:6] == "Frame ":
                frame_idx = int(f.split("Frame ")[1].split(":")[0])
                continue

            if f[:13] == "Total count: ":
                num_people = int(f.split("Total count: ")[1])
                continue
            
            # Frame data done in one line :))))))
            # We split each row at ":", then for each data we split again at "  ", then what's in between them would be the values we want
            # We put these values in an int constructor, and then put the whole loop in list comprehension, and cast it to a tuple
            t = tuple([int(f.split(":")[k].split("  ")[0]) for k in range(1,8)])
            frame_data_lis.append(t)
        
        data_dict[frame_idx] = (num_people, frame_rate, frame_data_lis)
    
    return data_dict