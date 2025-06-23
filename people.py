#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Example script to generate traffic in the simulation"""

import glob
import os
import sys
import time
import carla
import math
import random
import time
import queue
import numpy as np
import cv2
from pascal_voc_writer import Writer
import os
import dvs_api
import csv

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

from carla import VehicleLightState as vls

import argparse
import logging
from numpy import random

def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2, 3]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []

def build_projection_matrix(w, h, fov, is_behind_camera=False):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)

    if is_behind_camera:
        K[0, 0] = K[1, 1] = -focal
    else:
        K[0, 0] = K[1, 1] = focal

    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def get_image_point(loc, K, w2c):
        # Calculate 2D projection of 3D coordinate

        # Format the input coordinate (loc is a carla.Position object)
        point = np.array([loc.x, loc.y, loc.z, 1])
        # transform to camera coordinates
        point_camera = np.dot(w2c, point)

        # New we must change from UE4's coordinate system to an "standard"
        # (x, y ,z) -> (y, -z, x)
        # and we remove the fourth componebonent also
        point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

        # now project 3D->2D using the camera matrix
        point_img = np.dot(K, point_camera)
        # normalize
        point_img[0] /= point_img[2]
        point_img[1] /= point_img[2]

        return point_img[0:2]

def point_in_canvas(pos, img_h, img_w):
    """Return true if point is in canvas"""
    if (pos[0] >= 0) and (pos[0] < img_w) and (pos[1] >= 0) and (pos[1] < img_h):
        return True
    return False

def clip_bbox_to_screen(x_min, y_min, x_max, y_max, img_width, img_height):
    """
    Ensures the bounding box stays within the image frame.
    """
    x_min = max(0, min(img_width , x_min))
    y_min = max(0, min(img_height , y_min))
    x_max = max(0, min(img_width , x_max))
    y_max = max(0, min(img_height , y_max))
    return x_min, y_min, x_max, y_max

def dvs_callback(data): #store in image
    dvs_events = np.frombuffer(data.raw_data, dtype=np.dtype([
        ('x', np.uint16), ('y',np.uint16), ('t',np.int64), ('pol', np.bool)]))
    # data_dict['dvs_image'] = np.zeros((data.height, data.weight, 4), dtype=np.uint8)

    # dvs_img = np.zeros((data.height, data.width, 3), dtype=np.uint8)
    # dvs_img[dvs_events[:]['y'],dvs_events[:]['x'],dvs_events[:]['pol']*2] = 255

    # print(dvs_events[0]['t'], dvs_events[-1]['t'], max(dvs_events[:]['t']), min(dvs_events[:]['t']))
    # cv2.imwrite(f'dvs_output/{data.frame}.png', dvs_img)
    return dvs_events
def main():
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-n', '--number-of-vehicles',
        metavar='N',
        default=30,
        type=int,
        help='Number of vehicles (default: 30)')
    argparser.add_argument(
        '-w', '--number-of-walkers',
        metavar='W',
        default=10,
        type=int,
        help='Number of walkers (default: 10)')
    argparser.add_argument(
        '--safe',
        action='store_true',
        help='Avoid spawning vehicles prone to accidents')
    argparser.add_argument(
        '--filterv',
        metavar='PATTERN',
        default='vehicle.*',
        help='Filter vehicle model (default: "vehicle.*")')
    argparser.add_argument(
        '--generationv',
        metavar='G',
        default='All',
        help='restrict to certain vehicle generation (values: "1","2","All" - default: "All")')
    argparser.add_argument(
        '--filterw',
        metavar='PATTERN',
        default='walker.pedestrian.*',
        help='Filter pedestrian type (default: "walker.pedestrian.*")')
    argparser.add_argument(
        '--generationw',
        metavar='G',
        default='2',
        help='restrict to certain pedestrian generation (values: "1","2","All" - default: "2")')
    argparser.add_argument(
        '--tm-port',
        metavar='P',
        default=8000,
        type=int,
        help='Port to communicate with TM (default: 8000)')
    argparser.add_argument(
        '--asynch', 
        action='store_true',
        help='Activate asynchronous mode execution')
    argparser.add_argument(
        '--hybrid',
        action='store_true',
        help='Activate hybrid mode for Traffic Manager')
    argparser.add_argument(
        '-s', '--seed',
        metavar='S',
        type=int,
        help='Set random device seed and deterministic mode for Traffic Manager')
    argparser.add_argument(
        '--seedw',
        metavar='S',
        default=0,
        type=int,
        help='Set the seed for pedestrians module')
    argparser.add_argument(
        '--car-lights-on',
        action='store_true',
        default=False,
        help='Enable automatic car light management')
    argparser.add_argument(
        '--hero',
        action='store_true',
        default=False,
        help='Set one of the vehicles as hero')
    argparser.add_argument(
        '--respawn',
        action='store_true',
        default=False,
        help='Automatically respawn dormant vehicles (only in large maps)')
    argparser.add_argument(
        '--no-rendering',
        action='store_true',
        default=False,
        help='Activate no rendering mode')
    argparser.add_argument(
        '--night',
        action='store_true',
        default=False,
        help='Activate to set Nightmode')
    args = argparser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    vehicles_list = []
    walkers_list = []
    all_id = []
    client = carla.Client('localhost', 2000)
    client.set_timeout(120.0)
    synchronous_master = False
    random.seed(args.seed if args.seed is not None else int(time.time()))

    try:
        world = client.get_world()
        bp_lib = world.get_blueprint_library()

        traffic_manager = client.get_trafficmanager(args.tm_port)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        if args.respawn:
            traffic_manager.set_respawn_dormant_vehicles(True)
        if args.hybrid:
            traffic_manager.set_hybrid_physics_mode(True)
            traffic_manager.set_hybrid_physics_radius(70.0)
        if args.seed is not None:
            traffic_manager.set_random_device_seed(args.seed)

        settings = world.get_settings()
        if not args.asynch:
            traffic_manager.set_synchronous_mode(True)
            if not settings.synchronous_mode:
                synchronous_master = True
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.0333
            else:
                synchronous_master = False
        else:
            print("You are currently in asynchronous mode. If this is a traffic simulation, \
            you could experience some issues. If it's not working correctly, switch to synchronous \
            mode by using traffic_manager.set_synchronous_mode(True)")

        if args.no_rendering:
            settings.no_rendering_mode = True
        world.apply_settings(settings)

        blueprints = get_actor_blueprints(world, args.filterv, args.generationv)
        if not blueprints:
            raise ValueError("Couldn't find any vehicles with the specified filters")
        blueprintsWalkers = get_actor_blueprints(world, args.filterw, args.generationw)
        if not blueprintsWalkers:
            raise ValueError("Couldn't find any walkers with the specified filters")

        if args.safe:
            blueprints = [x for x in blueprints if x.get_attribute('base_type') == 'car']

        blueprints = sorted(blueprints, key=lambda bp: bp.id)

        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        if args.number_of_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif args.number_of_vehicles > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, args.number_of_vehicles, number_of_spawn_points)
            args.number_of_vehicles = number_of_spawn_points

        # @todo cannot import these directly.
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        # --------------
        # Spawn vehicles
        # --------------

        # Generate EGO vehicle
        ego_bp =bp_lib.find('vehicle.tesla.model3')
        ego = world.spawn_actor(ego_bp, random.choice(spawn_points))
        ego.set_autopilot(True)
        print("Ego vehicle is now on autopilot!")

        batch = []
        hero = args.hero
        for n, transform in enumerate(spawn_points):
            if n >= args.number_of_vehicles:
                break
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            if hero:
                blueprint.set_attribute('role_name', 'hero')
                hero = False
            else:
                blueprint.set_attribute('role_name', 'autopilot')

            # spawn the cars and set their autopilot and light state all together
            batch.append(SpawnActor(blueprint, transform)
                .then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))

        for response in client.apply_batch_sync(batch, synchronous_master):
            if response.error:
                logging.error(response.error)
            else:
                vehicles_list.append(response.actor_id)

        # Set automatic vehicle lights update if specified
        if args.car_lights_on:
            all_vehicle_actors = world.get_actors(vehicles_list)
            for actor in all_vehicle_actors:
                traffic_manager.update_vehicle_lights(actor, True)

        # -------------
        # Spawn Walkers
        # -------------
        # some settings
        percentagePedestriansRunning = 0.0      # how many pedestrians will run
        percentagePedestriansCrossing = 0.0     # how many pedestrians will walk through the road
        if args.seedw:
            world.set_pedestrians_seed(args.seedw)
            random.seed(args.seedw)
        # 1. take all the random locations to spawn
        spawn_points = []
        for i in range(args.number_of_walkers):
            spawn_point = carla.Transform()
            loc = world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        # 2. we spawn the walker object
        batch = []
        walker_speed = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(blueprintsWalkers)
            # set as not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            # set the max speed
            if walker_bp.has_attribute('speed'):
                if (random.random() > percentagePedestriansRunning):
                    # walking
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            else:
                print("Walker has no speed")
                walker_speed.append(0.0)
            batch.append(SpawnActor(walker_bp, spawn_point))
        results = client.apply_batch_sync(batch, True)
        walker_speed2 = []
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2
        # 3. we spawn the walker controller
        batch = []
        walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(walkers_list)):
            batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
        results = client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list[i]["con"] = results[i].actor_id
        # 4. we put together the walkers and controllers id to get the objects from their id
        for i in range(len(walkers_list)):
            all_id.append(walkers_list[i]["con"])
            all_id.append(walkers_list[i]["id"])
        all_actors = world.get_actors(all_id)

        # wait for a tick to ensure client receives the last transform of the walkers we have just created
        if args.asynch or not synchronous_master:
            world.wait_for_tick()
        else:
            world.tick()

        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(all_id), 2):
            # start walker
            all_actors[i].start()
            # set walk to random point
            all_actors[i].go_to_location(world.get_random_location_from_navigation())
            # max speed
            all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))

        print('spawned %d vehicles and %d walkers, press Ctrl+C to exit.' % (len(vehicles_list), len(walkers_list)))

        # Example of how to use Traffic Manager parameters
        traffic_manager.global_percentage_speed_difference(30.0)
        # sensor_list = ["sensor.camera.rgb", 
        #     "sensor.camera.semantic_segmentation",
        #     "sensor.camera.instance_segmentation",
        #     "sensor.camera.depth",
        #     "sensor.camera.optical_flow"]
        sensor_list = ["sensor.camera.rgb", "sensor.camera.dvs"]
        # image_queue_1 = queue.Queue() 
        # image_queue_2 = queue.Queue() 
        # image_queue_3 = queue.Queue()
        # image_queue_4 = queue.Queue()
        # image_queue_5 = queue.Queue()
        # image_queue_6 = queue.Queue()
        # image_queue_7 = queue.Queue()

        # Queue list
        rgb_queue_list = []
        for i in range(7):
            rgb_queue_list.append(queue.Queue())
        
        semantic_queue_list = []
        for i in range(7):
            semantic_queue_list.append(queue.Queue())

        instance_queue_list = []
        for i in range(7):
            instance_queue_list.append(queue.Queue())

        depth_queue_list = []
        for i in range(7):
            depth_queue_list.append(queue.Queue())

        optical_queue_list = []
        for i in range(7):
            optical_queue_list.append(queue.Queue())

        dvs_queue_list = []
        for i in range(7):
            dvs_queue_list.append(queue.Queue())

        total_queue_list = [rgb_queue_list, dvs_queue_list]
        spectator = world.get_spectator()
        
        rgb_camera_bp_list = []
        dvs_camera_bp_list = []

        total_camera_bp_list = [rgb_camera_bp_list, dvs_camera_bp_list]

        for j, sensor in enumerate(sensor_list):
            camera_bp_list = total_camera_bp_list[j]
            for i in range(7):
                camera_bp_list.append(bp_lib.find(sensor))
                if i < 5:
                    camera_bp_list[i].set_attribute("image_size_x", "1280")
                    camera_bp_list[i].set_attribute("image_size_y", "720")
                    camera_bp_list[i].set_attribute("fov", "70")
                if i == 5:
                    camera_bp_list[i].set_attribute("image_size_x", "1280")
                    camera_bp_list[i].set_attribute("image_size_y", "720")
                    camera_bp_list[i].set_attribute("fov", "110")
                if i == 6:
                    camera_bp_list[i].set_attribute("image_size_x", "1280")
                    camera_bp_list[i].set_attribute("image_size_y", "720")
       
        # Night Mode
        print("Night Mode:", args.night)
        if args.night:
            weather = carla.WeatherParameters(
                sun_altitude_angle=-90.0
            )
            world.set_weather(weather)
            # raw_camera_bp.set_attribute("positive_threshold", "1.3")
            # raw_camera_bp.set_attribute("negative_threshold", "1.3")


        # camera_init_trans = carla.Transform(carla.Location(z=1.5))
        camera_transform_1 = carla.Transform(
            carla.Location(x=0.5, y=0, z=1.5),
        )
        camera_transform_2 = carla.Transform(
            carla.Location(x=0.5, y=-0.6, z=1.5),
            carla.Rotation(pitch=0, yaw=-55, roll=0)
        )
        camera_transform_3 = carla.Transform(
            carla.Location(x=0.5, y=0.6, z=1.5),
            carla.Rotation(pitch=0, yaw=55, roll=0)
        )
        camera_transform_4 = carla.Transform(
            carla.Location(x=-0.5, y=-0.6, z=1.5),
            carla.Rotation(pitch=0, yaw=-110, roll=0)
        )
        camera_transform_5 = carla.Transform(
            carla.Location(x=-0.5, y=0.6, z=1.5),
            carla.Rotation(pitch=0, yaw=110, roll=0)
        )
        camera_transform_6 = carla.Transform(
            carla.Location(x=-1.2, y=0, z=1.5),
            carla.Rotation(pitch=0, yaw=180, roll=0)
        )
        camera_transform_7 = carla.Transform(
            carla.Location(z=20),
            carla.Rotation(pitch=-90, yaw=0, roll=0)
        )
        camera_transform_list = [camera_transform_1, camera_transform_2, camera_transform_3, camera_transform_4, camera_transform_5, camera_transform_6, camera_transform_7]
        
        rgb_camera_list = []
        dvs_camera_list = []

        total_camera_list = [rgb_camera_list, dvs_camera_list]

        for j, camera_list in enumerate(total_camera_list):
            camera_list.append(world.spawn_actor(total_camera_bp_list[j][i], camera_transform_7, attach_to=ego))

        # Get the world to camera matrix
        # world_2_camera = np.array(rgb_camera_list[0].get_transform().get_inverse_matrix())

        # Get the attributes from the camera
        image_w = 1280
        image_h = 720
        fov_1 = 70.0
        fov_2 = 110.0
        fov_3 = 90.0

        # negative_threshold = camera_bp.get_attribute("negative_threshold")

        # print("negative_threshold: ", negative_threshold)

        # Calculate the camera projection matrix to project from 3D -> 2D
        K_1 = build_projection_matrix(image_w, image_h, fov_1)
        K_2 = build_projection_matrix(image_w, image_h, fov_2)
        K_3 = build_projection_matrix(image_w, image_h, fov_3)
        # K_b = build_projection_matrix(image_w, image_h, fov_1, is_behind_camera=True)

        output_path = "output/"
        label_path = "labels"
        label_3d_path = "labels_3d"
        for sensor in sensor_list:
            for i in range(7):
                if not os.path.exists(output_path + sensor + f"/{i}"):
                    os.makedirs(output_path + sensor + f"/{i}")
                if sensor == "sensor.camera.dvs":
                    with open(output_path + sensor + f"/{i}" + "/dvs_output.csv", mode="w",  newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(['x', 'y', 't', 'pol'])
                        csvfile.close()
                    with open(output_path + sensor + f"/{i}" + "/bbox.csv" , "w", encoding = "utf-8") as file:
                        writer = csv.writer(file)
                        writer.writerow(['t', 'x', 'y', 'w', 'h', 'class_id','class_confidence'])
                        file.close()
            print("make dir: " + output_path + sensor)
        for i in range(7):
            if not os.path.exists(output_path + label_path + f"/{i}"):
                os.makedirs(output_path + label_path + f"/{i}")
        print("make dir: " + output_path + label_path)

        for i in range(7):
            if not os.path.exists(output_path + label_3d_path + f"/{i}"):
                os.makedirs(output_path + label_3d_path + f"/{i}")
        print("make dir: " + output_path + label_3d_path)
        # if not os.path.exists(output_path + image_path_2):
        #     os.makedirs(output_path + image_path_2)
        #     print("make dir: " + output_path + image_path_2)
        # if not os.path.exists(output_path + image_path_3):
        #     os.makedirs(output_path + image_path_3)
        #     print("make dir: " + output_path + image_path_3)
        # if not os.path.exists(output_path + image_path_4):
        #     os.makedirs(output_path + image_path_4)
        #     print("make dir: " + output_path + image_path_4)
        # if not os.path.exists(output_path + image_path_5):
        #     os.makedirs(output_path + image_path_5)
        #     print("make dir: " + output_path + image_path_5)
        # if not os.path.exists(output_path + image_path_6):
        #     os.makedirs(output_path + image_path_6)
        #     print("make dir: " + output_path + image_path_6)
        # if not os.path.exists(output_path + image_path_7):
        #     os.makedirs(output_path + image_path_7)
        #     print("make dir: " + output_path + image_path_7)
        # if not os.path.exists(output_path + rgb_label_path):
        #     os.makedirs(output_path + rgb_label_path)
        #     print("make dir: " + output_path + rgb_label_path)

        # dvs_output_path = "output/dvs_output.csv"
        # with open(dvs_output_path, mode="w",  newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow(['x', 'y', 't', 'pol'])
        #     file.close()

        # with open("output/bbox.csv", "w", encoding = "utf-8") as file:
        #     writer = csv.writer(file)
        #     writer.writerow(['t', 'x', 'y', 'w', 'h', 'class_id','class_confidence'])
        #     file.close()
        # world.tick()
        # raw_camera.listen(lambda data: time_queue.put(dvs_api.dvs_callback_csv(data, dvs_output_path)))

        # for j, camera_list in enumerate(total_camera_list):
        #     image_queue_list = total_queue_list[j]
        #     for i, camera in enumerate(camera_list):
        #         camera.listen(image_queue_list[i].put)
        world.tick()
        for j, camera_list in enumerate(total_camera_list):
            image_queue_list = total_queue_list[j]
            for i, camera in enumerate(camera_list):
                camera.listen(image_queue_list[i].put)

                # if j == 0:
                #     camera.listen(image_queue_list[i].put)
                # if j == 1:
                #     dvs_output_path = output_path + sensor + f"/{i}"
                #     camera.listen(lambda data, index=i: 
                #                 image_queue_list[index].put(dvs_events(data)))
        if args.asynch or not synchronous_master:
            world.wait_for_tick()
        else:
            world.tick()
        # camera_1.listen(image_queue_1.put)
        # camera_2.listen(image_queue_2.put)
        # camera_3.listen(image_queue_3.put)
        # camera_4.listen(image_queue_4.put)
        # camera_5.listen(image_queue_5.put)
        # camera_6.listen(image_queue_6.put)
        # camera_7.listen(image_queue_7.put)
        edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]

        print("Queue empty before main loop?", dvs_queue_list[0].empty())
        times = 0
        while True:
            # print(times)
            times += 1
            if not args.asynch and synchronous_master:
                world.tick()
            else:
                world.wait_for_tick()
            if dvs_queue_list[0].empty():
                print("Queue empty, ERROR")

            # world_2_camera_0 = np.array(rgb_camera_list[0].get_transform().get_inverse_matrix())
            # world_2_camera_1 = np.array(rgb_camera_list[1].get_transform().get_inverse_matrix())
            # world_2_camera_2 = np.array(rgb_camera_list[2].get_transform().get_inverse_matrix())
            # world_2_camera_3 = np.array(rgb_camera_list[3].get_transform().get_inverse_matrix())
            # world_2_camera_4 = np.array(rgb_camera_list[4].get_transform().get_inverse_matrix())
            # world_2_camera_5 = np.array(rgb_camera_list[5].get_transform().get_inverse_matrix())
            # world_2_camera_6 = np.array(rgb_camera_list[6].get_transform().get_inverse_matrix())

            bboxes_0 = []
            bboxes_1 = []
            bboxes_2 = []
            bboxes_3 = []
            bboxes_4 = []
            bboxes_5 = []
            bboxes_6 = []
            bboxes_list = [bboxes_0, bboxes_1, bboxes_2, bboxes_3, bboxes_4, bboxes_5, bboxes_6]

            # bboxes_3d_0 = []
            # bboxes_3d_1 = []
            # bboxes_3d_2 = []
            # bboxes_3d_3 = []
            # bboxes_3d_4 = []
            # bboxes_3d_5 = []
            # bboxes_3d_6 = []
            # bboxes_3d_list = [bboxes_3d_0, bboxes_3d_1, bboxes_3d_2, bboxes_3d_3, bboxes_3d_4, bboxes_3d_5, bboxes_3d_6]

            for npc in world.get_actors().filter('*vehicle*'):
                if npc.id != ego.id:
                    bb = npc.bounding_box
                    dist = npc.get_transform().location.distance(ego.get_transform().location)
            #         x = npc.get_velocity().x
            #         y = npc.get_velocity().y
            #         z = npc.get_velocity().z
            #         velocity = (x**2+y**2+z**2)**0.5
                    if dist < 50:
                        for i in range(7):
                            forward_vec = rgb_camera_list[i].get_transform().get_forward_vector()
                            ray = npc.get_transform().location - rgb_camera_list[i].get_transform().location
                            if forward_vec.dot(ray) > 0:
                                # p1 = get_image_point(bb.location, K, world_2_camera)
                                verts = [v for v in bb.get_world_vertices(npc.get_transform())]
                                x_max = -10000
                                x_min = 10000
                                y_max = -10000
                                y_min = 10000
                                if i == 5:
                                    K = K_2
                                elif i == 6:
                                    K = K_3
                                else:
                                    K = K_1
                                world_2_camera = np.array(rgb_camera_list[i].get_transform().get_inverse_matrix())
                                for vert in verts:
                                  # Modify data (example: append a new row)
                                    p = get_image_point(vert, K, world_2_camera)
                                    if p[0] > x_max:
                                        x_max = p[0]
                                    if p[0] < x_min:
                                        x_min = p[0]
                                    if p[1] > y_max:
                                        y_max = p[1]
                                    if p[1] < y_min:
                                        y_min = p[1]
                                # if x_min < 0 and x_max < image_w:
                                #     x_min = 0
                                # if x_max > image_w and x_min > 0:
                                #     x_max = image_w
                                # if y_min < 0 and y_max < image_h:
                                #     y_min = 0
                                # if y_max > image_h and y_min > 0:
                                #     y_max = image_h
                                x_min, y_min, x_max, y_max = clip_bbox_to_screen(x_min, y_min, x_max, y_max, image_w, image_h)
                                # Add the object to the frame (ensure it is inside the image)
                                if x_max - x_min > 0 and x_max - x_min < image_w and y_max - y_min > 0 and y_max - y_min < image_h: 
                                    center_x = (x_min + x_max)/2
                                    center_y = (y_min + y_max)/2
                                    w = (x_max - x_min)
                                    w_normal = (x_max - x_min)/image_w
                                    h = (y_max - y_min)
                                    h_normal = (y_max - y_min)/image_h
                                    x_normal = center_x/image_w
                                    y_normal = center_y/image_h
                                # if w_normal > 0 and h_normal > 0:
                                    bboxes_list[i].append(('0', x_normal, y_normal, w_normal, h_normal))

                                # temp_bbox = []
                                # for edge in edges:
                                #     p1 = get_image_point(verts[edge[0]], K, world_2_camera)
                                #     p2 = get_image_point(verts[edge[1]],  K, world_2_camera)

                                #     p1_in_canvas = point_in_canvas(p1, image_h, image_w)
                                #     p2_in_canvas = point_in_canvas(p2, image_h, image_w)

                                #     if not p1_in_canvas and not p2_in_canvas:
                                #         continue

                                #     ray0 = verts[edge[0]] - camera.get_transform().location
                                #     ray1 = verts[edge[1]] - camera.get_transform().location
                                #     cam_forward_vec = camera.get_transform().get_forward_vector()

                                #     # One of the vertex is behind the camera
                                #     if not (cam_forward_vec.dot(ray0) > 0):
                                #         p1 = get_image_point(verts[edge[0]], K_b, world_2_camera)
                                #     if not (cam_forward_vec.dot(ray1) > 0):
                                #         p2 = get_image_point(verts[edge[1]], K_b, world_2_camera)
                                #     temp_bbox.append(((int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1]))))
                                # if temp_bbox != []:
                                #     bboxes_3d_list[i].append(('0', temp_bbox))
                                # bboxes_3d_list[i].append(('0', temp_bbox))


            for npc in world.get_actors().filter('*pedestrian*'):
            #     # if npc.id != vehicle.id:
                    bb = npc.bounding_box
                    dist = npc.get_transform().location.distance(ego.get_transform().location)
                    if dist < 50:
                        for i in range(7):
                            forward_vec = rgb_camera_list[i].get_transform().get_forward_vector()
                            ray = npc.get_transform().location - ego.get_transform().location
                            if forward_vec.dot(ray) > 0:
                                # p1 = get_image_point(bb.location, K, world_2_camera)
                                verts = [v for v in bb.get_world_vertices(npc.get_transform())]
                                x_max = -10000
                                x_min = 10000
                                y_max = -10000
                                y_min = 10000
                                if i == 5:
                                    K = K_2
                                elif i == 6:
                                    K = K_3
                                else:
                                    K = K_1
                                world_2_camera = np.array(rgb_camera_list[i].get_transform().get_inverse_matrix())
                                for vert in verts:
                                    p = get_image_point(vert, K, world_2_camera)
                                    if p[0] > x_max:
                                        x_max = p[0]
                                    if p[0] < x_min:
                                        x_min = p[0]
                                    if p[1] > y_max:
                                        y_max = p[1]
                                    if p[1] < y_min:
                                        y_min = p[1]

                                # Add the object to the frame (ensure it is inside the image)
                                if x_min > 0 and x_max < image_w and y_min > 0 and y_max < image_h: 
                                    center_x = (x_min + x_max)/2
                                    center_y = (y_min + y_max)/2
                                    w = (x_max - x_min)
                                    h = (y_max - y_min)
                                    w_normal = (x_max - x_min)/image_w
                                    h_normal = (y_max - y_min)/image_h
                                    x_normal = center_x/image_w
                                    y_normal = center_y/image_h

                                    bboxes_list[i].append(('1', x_normal, y_normal, w_normal, h_normal))
                                
                                # temp_bbox = []
                                # for edge in edges:
                                #     p1 = get_image_point(verts[edge[0]], K, world_2_camera)
                                #     p2 = get_image_point(verts[edge[1]],  K, world_2_camera)

                                #     p1_in_canvas = point_in_canvas(p1, image_h, image_w)
                                #     p2_in_canvas = point_in_canvas(p2, image_h, image_w)

                                #     if not p1_in_canvas and not p2_in_canvas:
                                #         continue

                                #     ray0 = verts[edge[0]] - camera.get_transform().location
                                #     ray1 = verts[edge[1]] - camera.get_transform().location
                                #     cam_forward_vec = camera.get_transform().get_forward_vector()

                                #     # One of the vertex is behind the camera
                                #     if not (cam_forward_vec.dot(ray0) > 0):
                                #         p1 = get_image_point(verts[edge[0]], K_b, world_2_camera)
                                #     if not (cam_forward_vec.dot(ray1) > 0):
                                #         p2 = get_image_point(verts[edge[1]], K_b, world_2_camera)
                                #     temp_bbox.append(((int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1]))))
                                # if temp_bbox != []:
                                #     bboxes_3d_list[i].append(('1', temp_bbox))

            # # Save the bounding boxes in the scene
            rgb_output_list = []
            semantic_output_list = []
            instance_output_list = []
            depth_output_list = []
            opitcal_output_list = []
            dvs_output_list = []
            # output_list = [rgb_output_list, semantic_output_list, instance_output_list, depth_output_list, opitcal_output_list]
            output_list = [rgb_output_list, dvs_output_list]

            for i, image_queue_list in enumerate(total_queue_list):
                # image_output_list = output_list[i]
                for img_queue in image_queue_list:
                    output_list[i].append(img_queue.get())
                    # if img_queue.empty() != True:
                    #     output_list[i].append(img_queue.get())

            frame_path = '%06d' % output_list[0][0].frame
            for j, sensor in enumerate(sensor_list):
                for i in range(7):
                    # if j == 1:  # semantic
            #             output_list[j][i].save_to_disk(output_path + sensor + f"/{i}/" + frame_path + '.png', carla.ColorConverter.CityScapesPalette)                    
            #         elif j ==3:   # depth
            #             output_list[j][i].save_to_disk(output_path + sensor + f"/{i}/" + frame_path + '.png', carla.ColorConverter.LogarithmicDepth)
            #         elif j == 4:  # optical flow
            #             opt_image = output_list[j][i].get_color_coded_flow()
            #             flow_array = np.frombuffer(opt_image.raw_data, dtype=np.uint8)
            #             flow_array = flow_array.reshape((opt_image.height, opt_image.width, 4))
            #             flow_array = flow_array[:, :, :3] 
                        # cv2.imwrite(output_path + sensor + f"/{i}/" + frame_path + '.png', cv2.cvtColor(flow_array, cv2.COLOR_RGB2BGR))
            #         else:
                    if j == 0:
                        output_list[j][i].save_to_disk(output_path + sensor + f"/{i}/" + frame_path + '.png')
                    elif j == 1:
                        dvs_events = np.frombuffer(output_list[j][i].raw_data, dtype=np.dtype([
                            ('x', np.uint16), ('y',np.uint16), ('t',np.int64), ('pol', np.bool)]))
                        dvs_img = np.zeros((image_h, image_w, 3), dtype=np.uint8)
                        dvs_img[dvs_events[:]['y'],dvs_events[:]['x'],dvs_events[:]['pol']*2] = 255
                        cv2.imwrite(output_path + sensor + f"/{i}/" + frame_path + '.png', dvs_img)

                        dvs_event_copy = dvs_events.copy()
                        dvs_event_copy['t'] //= 1000
                        with open(output_path + sensor + f"/{i}" + "/dvs_output.csv", mode="a",  newline='') as file:
                            writer = csv.writer(file)
                            for event in dvs_event_copy:
                                writer.writerow(event)
                            file.close()
                        timestamp = dvs_event_copy[0]['t']
                        bboxes = bboxes_list[i]
                        with open(output_path + sensor + f"/{i}" + "/bbox.csv", "a", encoding = "utf-8") as file:
                            writer = csv.writer(file)
                            for bbox in bboxes:
                                event = [timestamp] + [bbox[1], bbox[2], bbox[3], bbox[4], int(bbox[0]), 1.0]
                                writer.writerow(event)
                            file.close()
            # image_1.save_to_disk(output_path + image_path_1 + frame_path + '.png') 
            # image_2.save_to_disk(output_path + image_path_2 + frame_path + '.png')
            # image_3.save_to_disk(output_path + image_path_3 + frame_path + '.png')
            # image_4.save_to_disk(output_path + image_path_4 + frame_path + '.png')
            # image_5.save_to_disk(output_path + image_path_5 + frame_path + '.png')
            # image_6.save_to_disk(output_path + image_path_6 + frame_path + '.png')
            # image_7.save_to_disk(output_path + image_path_7 + frame_path + '.png')
            for i in range(7):
                bboxes = bboxes_list[i]
                with open( output_path + label_path + f"/{i}/" + frame_path+".txt", "w", encoding = "utf-8") as file:
                    for bbox in bboxes:
                        # print(len(bbox))
                        # if len(bbox) == 5:
                        file.write(bbox[0]+f" {bbox[1]} {bbox[2]} {bbox[3]} {bbox[4]}\n")
                        # else:
                            # continue
                    file.close()

                
                # bboxes_3d = bboxes_3d_list[i]
                # with open( output_path + label_3d_path + f"/{i}/" + frame_path+".txt", "w", encoding = "utf-8") as file3d:
                #     for bbox in bboxes_3d:
                #         # print(len(bbox))
                #         # if len(bbox) == 5:
                #         file3d.write(f"{bbox}\n")
                #         # else:
                #             # continue
                #     file3d.close()
                # if total_queue_list[1][i].empty() != True:
                #     # print(output_list[1])
                #     # print(i)
                #     timestamp = total_queue_list[1][i].get()
                    # with open(output_path + sensor + f"/{i}" + "/bbox.csv", "a", encoding = "utf-8") as file:
                    #     writer = csv.writer(file)
                    #     for bbox in bboxes:
                    #         event = [timestamp] + [bbox[1], bbox[2], bbox[3], bbox[4], int(bbox[0]), 1.0]
                    #         writer.writerow(event)
                    #     file.close()

            if times % 300 == 0:
                print("times = ", times)
            if times > 300:
                for camera_list in total_camera_list:
                    for camera in camera_list:
                        camera.stop()
                print("Finish")
                break
    finally:

        if not args.asynch and synchronous_master:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.no_rendering_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)

        print('\ndestroying %d vehicles' % len(vehicles_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

        # stop walker controllers (list is [controller, actor, controller, actor ...])
        for i in range(0, len(all_id), 2):
            all_actors[i].stop()

        print('\ndestroying %d walkers' % len(walkers_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in all_id])

        time.sleep(0.5)

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')