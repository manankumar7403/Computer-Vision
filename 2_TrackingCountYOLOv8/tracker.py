import math

class Tracker:
    def __init__(self):                                       # init the class, called when instance of the class is created
        # Store the center positions of the tracked objects(keys -> object IDs, values -> tuples representing (x,y) coordinates)
        self.center_points = {}                           
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0                                     # keeps track of the unique IDs


    def update(self, objects_rect):            # defines the update method -> takes a list of object rectangles as input
        # Objects boxes and ids
        objects_bbs_ids = []                   # stores the bounding boxes and IDs of the tracked objects

        # Get center point of new object
        for rect in objects_rect:              # iterates over the rectangles in objects_rect
            x, y, w, h = rect                  # unpacks the rectangle coordinates (x,y,width,height)
            cx = (x + x + w) // 2              # Calculates the x-coordinate of the center of the rectangle
            cy = (y + y + h) // 2              # (x + x + w) // 2: This calculates the average x-coordinate of the rectangle. The (x + x + w) part represents the sum of the leftmost x-coordinate (x) and the rightmost x-coordinate (x + w). Dividing this sum by 2 (// 2) gives the average x-coordinate. (y + y + h) // 2: Similarly, this calculates the average y-coordinate of the rectangle. The (y + y + h) part represents the sum of the topmost y-coordinate (y) and the bottom y-coordinate (y + h). Dividing this sum by 2 (// 2) gives the average y-coordinate.

            # Find out if that object was detected already
            same_object_detected = False                        # checks if the object has already been detected or not
            for id, pt in self.center_points.items():           # id -> object id, pt -> assigned the value of a tuple(representing (cx,cy) coordinates)
                dist = math.hypot(cx - pt[0], cy - pt[1])       # calculates the euclidean distance between the center of the current bounding box(cx,cy) and the center of the object stored in self_center_points. -> resulting Euclidean Distance between the 2 points representing how far apart they are in coordinate space

                if dist < 35:                                   # whether the calculated distance dist b/w center of curr bounding box and center of tracked obj is less than 35. This dist threshold(35) determines if the curr detection corresponds to the same object that was previously tracked. If the distance is below this threshold, it suggests that same obj detected again.
                    self.center_points[id] = (cx, cy)           # curr detection true, it means the curr detection corresponds to the same object. Center coordinates of the tracked object stored in self_center_points, updated with new center coordinates(cx,cy) -> ensures tracking info is up-to-date.
                    #print(self.center_points)
                    objects_bbs_ids.append([x, y, w, h, id])    # bounding box coordinates ([x, y, w, h]) and the object ID (id) are appended to the objects_bbs_ids list. This list is used to keep track of bounding boxes and their corresponding IDs.
                    same_object_detected = True                 #  set to True to indicate that the current detection corresponds to a tracked object.
                    break                                       # exits the loop when once a match is found, since the current detection has been associated with a tracked object, no need to continue checking the remaining objs

            # New object is detected we assign the ID to that object
            if same_object_detected is False:                   # current detection not close to any previously tracked objects         
                self.center_points[self.id_count] = (cx, cy)    # current detection is of a new object, new entry is added to self.center_points using the current value self.id_count as obj id. Center coordinates of new object are (cx,cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])  # bounding box coordinates ([x, y, w, h]) and the newly assigned object ID (self.id_count) are appended to the objects_bbs_ids list. This list is used to keep track of bounding boxes and their corresponding IDs. 
                self.id_count += 1                               # ensures that the next newly detected object will be assigned a unique ID, preventing overlap with existing IDs.

        # Clean the dict by center points to remove IDS not used anymore
        # The purpose of these lines is to clean up the self.center_points dictionary by removing entries corresponding to IDs that are no longer used or associated with any detected object. 
        new_center_points = {}                                   # to store the center points of currently tracked objects, will be updated to only include the IDs that are still in use
        for obj_bb_id in objects_bbs_ids:                        # iterate through the list objects_bbs_ids, which contains information about currently detected objects, including their bounding box coordinates and IDs.
            _, _, _, _, object_id = obj_bb_id                    # Unpacks the information from the current obj_bb_id. The bounding box coordinates are ignored (_ placeholder), and only the object_id is extracted.
            center = self.center_points[object_id]               # retrieving the center coordinates of the object with the specified object_id from the self.center_points dictionary.
            new_center_points[object_id] = center                # updating the new_center_points dictionary with the center coordinates of the currently tracked object

        # Update dict with IDs not used -> removed
        self.center_points = new_center_points.copy()            # replace og self.center_points dict with a copy of new_center_points dict. Now self.center_points dict holds the cleaned and updated info about the currently tracked objects
        return objects_bbs_ids                                   # contains information about the bounding box coordinates and associated IDs of the detected objects. This list is used by the main tracking logic to update the state of tracked objects in test.py