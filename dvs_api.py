import carla
import csv
import numpy as np
import time
import cv2
import os

# client = carla.Client('localhost', 2000)
# world = client.get_world()

# # Retrieve the spectator object
# spectator = world.get_spectator()

# # Get the location and rotation of the spectator through its transform
# transform = spectator.get_transform()

# location = transform.location
# rotation = transform.rotation
# # print(location, rotation)

# # Set the spectator to a position
# spectator.set_transform(carla.Transform(carla.Location(x=-62.010975, y=1.288139, z=17.589510),carla.Rotation(pitch=-48.045647, yaw=53.329460, roll=0.000444)))

# camera_trans = carla.Transform(carla.Location(z=0.1))

# dvs_camera_bp= world.get_blueprint_library().find('sensor.camera.dvs')

# dvs_camera = world.spawn_actor(dvs_camera_bp, camera_trans, attach_to=spectator)

# os.makedirs("dvs_output")

def dvs_callback_img(data): #store in image
    dvs_events = np.frombuffer(data.raw_data, dtype=np.dtype([
        ('x', np.uint16), ('y',np.uint16), ('t',np.int64), ('pol', np.bool)]))
    # data_dict['dvs_image'] = np.zeros((data.height, data.weight, 4), dtype=np.uint8)
    dvs_img = np.zeros((data.height, data.width, 3), dtype=np.uint8)
    dvs_img[dvs_events[:]['y'],dvs_events[:]['x'],dvs_events[:]['pol']*2] = 255
    # print(dvs_events[0]['t'], dvs_events[-1]['t'], max(dvs_events[:]['t']), min(dvs_events[:]['t']))
    # cv2.imwrite(f'dvs_output/{data.frame}.png', dvs_img)
    return dvs_img

def dvs_callback_csv(data, dvs_output_path): # store in csv file
    # print("length = ",len(data))

    dvs_events = np.frombuffer(data.raw_data, dtype=np.dtype([
        ('x', np.uint16), ('y',np.uint16), ('t',np.int64), ('pol', np.bool)]))
    dvs_event_copy = dvs_events.copy()
    dvs_event_copy['t'] //= 1000
    # print("dvs_events: ", dvs_events)
    # print("dvs_event_copy: ", dvs_event_copy)

    with open(dvs_output_path, mode="a",  newline='') as file:
        writer = csv.writer(file)
        for event in dvs_event_copy:
            # combine = event.tolist()

            # print("event: ", event)
            writer.writerow(event)
        file.close()
    return dvs_event_copy[0]['t']
    
# dvs_camera.listen(lambda DVSEventArray: dvs_callback_img(DVSEventArray))
# time.sleep(3)
# dvs_camera.stop()
# print("STOP")
