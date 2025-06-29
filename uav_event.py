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
# import sys
# sys.path.append("/home/apg/carla/PythonAPI/carla/dist/carla-0.9.14-py3.7-linux-x86_64.egg")
# import carla
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

def lerp(a, b, alpha):
    return a + alpha * (b - a)
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
    argparser.add_argument(
        '-i', '--intersection',
        metavar='I',
        default=1,
        type=int,
        help='Intersection from 1 to 4')
    args = argparser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    vehicles_list = []
    walkers_list = []
    all_id = []
    client = carla.Client('localhost', 2000)
    client.set_timeout(120.0)
    synchronous_master = False
    random.seed(args.seed if args.seed is not None else int(time.time()))
    corners = [
        carla.Location(x=-47, y=-62),
        carla.Location(x=106, y=-62),
        carla.Location(x=106, y=135),
        carla.Location(x=-47, y=135)
    ]
    height = 20
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
        # print(carla.__version__)
        # --------------
        # Spawn vehicles
        # --------------
        # Generate EGO vehicle
        # ego_bp =bp_lib.find('vehicle.tesla.model3')
        # tf = random.choice(world.get_map().get_spawn_points())
        # ego = world.try_spawn_actor(ego_bp, tf)
        # waypoint = world.get_map().get_waypoint(tf.location)
        # ego_transform = carla.Transform(carla.Location(x=waypoint.transform.location.x, y=waypoint.transform.location.y, z=-1.5), 
        #                                             carla.Rotation(pitch = 0, yaw=waypoint.transform.rotation.yaw, roll=0))
        # ego.set_transform(ego_transform)
        # traffic_manager.vehicle_percentage_speed_difference(ego, -1000.0)
        # traffic_manager.ignore_lights_percentage(ego, 100)

        # ego.set_autopilot(True)
        # print("Ego vehicle is now on autopilot!")

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
        image_queue = queue.Queue() 
        time_queue = queue.Queue()

        spectator = world.get_spectator()
        transform = spectator.get_transform()
        # location = transform.location
        # rotation = transform.rotation


        # spawn camera
        camera_bp = bp_lib.find('sensor.camera.rgb')
        # dvs_camera_bp = bp_lib.find('sensor.camera.dvs')
        raw_camera_bp = bp_lib.find('sensor.camera.dvs')

        # raw_camera_bp.set_attribute("image_size_x", "3840")
        # raw_camera_bp.set_attribute("image_size_y", "2160")
        # camera_bp.set_attribute("image_size_x", "3840")
        # camera_bp.set_attribute("image_size_y", "2160")

        raw_camera_bp.set_attribute("image_size_x", "1280")
        raw_camera_bp.set_attribute("image_size_y", "720")
        camera_bp.set_attribute("image_size_x", "1280")
        camera_bp.set_attribute("image_size_y", "720")

        # Night Mode
        print("Night Mode:", args.night)
        if args.night:
            weather = carla.WeatherParameters(
                sun_altitude_angle=-90.0
            )
            world.set_weather(weather)
            # raw_camera_bp.set_attribute("positive_threshold", "1.3")
            # raw_camera_bp.set_attribute("negative_threshold", "1.3")
            raw_camera_bp.set_attribute("enable_postprocess_effects", "false")
            # raw_camera_bp.set_attribute("noise_seed", "10")

            # # raw_camera_bp.set_attribute("negative_threshold", "2.3")
            # raw_camera_bp.set_attribute("refractory_period_ns", "100000")
            # raw_camera_bp.set_attribute('blur_amount', '1.5')
            # raw_camera_bp.set_attribute('blur_radius', '5')


        camera_init_trans = carla.Transform(carla.Location(z=0))

        camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=spectator)
        # dvs_camera = world.spawn_actor(dvs_camera_bp, camera_init_trans, attach_to=spectator)
        raw_camera = world.spawn_actor(raw_camera_bp, camera_init_trans, attach_to=spectator)


        # Get the world to camera matrix
        world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

        # Get the attributes from the camera
        image_w = raw_camera_bp.get_attribute("image_size_x").as_int()
        image_h = raw_camera_bp.get_attribute("image_size_y").as_int()
        fov = raw_camera_bp.get_attribute("fov").as_float()

        positive_threshold = raw_camera_bp.get_attribute("positive_threshold")
        negative_threshold = raw_camera_bp.get_attribute("negative_threshold")

        print("positive_threshold: ", positive_threshold)
        print("negative_threshold: ", negative_threshold)

        # Calculate the camera projection matrix to project from 3D -> 2D
        K = build_projection_matrix(image_w, image_h, fov)
        K_b = build_projection_matrix(image_w, image_h, fov, is_behind_camera=True)

        # Retrieve the first image
        # world.tick()
        # world.wait_for_tick()

        output_path = "output/"
        image_path = "images/"
        rgb_label_path = "rgb_labels/"
        # event_path = "events/"
        # dvs_label_path = "dvs_labels/"

        if not os.path.exists(output_path + image_path):
            os.makedirs(output_path + image_path)
            print("make dir: " + output_path + image_path)
        if not os.path.exists(output_path + rgb_label_path):
            os.makedirs(output_path + rgb_label_path)
            print("make dir: " + output_path + rgb_label_path)
        # if not os.path.exists(output_path + event_path):
        #     os.makedirs(output_path + event_path)
        #     print("make dir: " + output_path + event_path)
        # if not os.path.exists(output_path + dvs_label_path):
        #     os.makedirs(output_path + dvs_label_path)
        #     print("make dir: " + output_path + dvs_label_path)

        dvs_output_path = "output/dvs_output.csv"
        with open(dvs_output_path, mode="w",  newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['x', 'y', 't', 'pol'])
            file.close()

        with open("output/bbox.csv", "w", encoding = "utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(['t', 'x', 'y', 'w', 'h', 'class_id','class_confidence'])
            file.close()
        # world.tick()
        # raw_camera.listen(lambda data: time_queue.put(dvs_api.dvs_callback_csv(data, dvs_output_path)))
        spectator.set_transform(carla.Transform(carla.Location(x=-47, y=-62, z = height), carla.Rotation(pitch=-30, yaw=0)))

        raw_camera.listen(time_queue.put)
        camera.listen(image_queue.put)

        world.tick()
        times = 0
        index = 0
        while True:
            point1 = np.array([corners[index % 4].x, corners[index % 4].y])
            point2 = np.array([corners[(index + 1) % 4].x, corners[(index + 1) % 4].y])
            # print(point1)
            # print(point2)
            for step in range(500):
                alpha = step / 499
                pos = lerp(point1, point2, alpha)
                uav_location = carla.Location(x=pos[0], y=pos[1], z=height)
                yaw = np.degrees(np.arctan2(point2[1]-point1[1], point2[0]-point1[0]))
                spectator.set_transform(carla.Transform(uav_location, carla.Rotation(pitch=-30, yaw=yaw)))
                # print("camera: ", camera.get_transform().location)
                # print("spectator: ", spectator.get_transform().location)
                times += 1
                if not args.asynch and synchronous_master:
                    world.tick()
                else:
                    world.wait_for_tick()                    
                # spectator.set_transform(camera.get_transform())
                world_2_camera = np.array(camera.get_transform().get_inverse_matrix())
                bboxes = []
                bboxes_dvs = []
                for npc in world.get_actors().filter('*vehicle*'):
                    # if npc.id != vehicle.id:
                        bb = npc.bounding_box
                        dist = npc.get_transform().location.distance(camera.get_transform().location)
                        x = npc.get_velocity().x
                        y = npc.get_velocity().y
                        z = npc.get_velocity().z
                        velocity = (x**2+y**2+z**2)**0.5
                        if dist < 70:
                            forward_vec = camera.get_transform().get_forward_vector()
                            ray = npc.get_transform().location - camera.get_transform().location
                            if forward_vec.dot(ray) > 0:
                                p1 = get_image_point(bb.location, K, world_2_camera)
                                verts = [v for v in bb.get_world_vertices(npc.get_transform())]
                                x_max = -10000
                                x_min = 10000
                                y_max = -10000
                                y_min = 10000
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
                                    w_normal = (x_max - x_min)/image_w
                                    h = (y_max - y_min)
                                    h_normal = (y_max - y_min)/image_h
                                    x_normal = center_x/image_w
                                    y_normal = center_y/image_h

                                    bboxes.append(('0', x_normal, y_normal, w_normal, h_normal))
                                    if velocity >=0.1:
                                        bboxes_dvs.append([x_min, y_min, w, h, 2, 1.0])
                # world.tick()
                # world.wait_for_tick()

                for npc in world.get_actors().filter('*pedestrian*'):
                    # if npc.id != vehicle.id:
                        bb = npc.bounding_box
                        dist = npc.get_transform().location.distance(camera.get_transform().location)
                        # x = npc.get_velocity().x
                        # y = npc.get_velocity().y
                        # z = npc.get_velocity().z
                        # velocity = (x**2+y**2+z**2)**0.5
                        if dist < 60:
                            forward_vec = camera.get_transform().get_forward_vector()
                            ray = npc.get_transform().location - camera.get_transform().location
                            if forward_vec.dot(ray) > 0:
                                p1 = get_image_point(bb.location, K, world_2_camera)
                                verts = [v for v in bb.get_world_vertices(npc.get_transform())]
                                x_max = -10000
                                x_min = 10000
                                y_max = -10000
                                y_min = 10000
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

                                    bboxes.append(('1', x_normal, y_normal, w_normal, h_normal))
                                    # if velocity > 0:
                                    bboxes_dvs.append([x_min, y_min , w, h, 0, 1.0])
                # Save the bounding boxes in the scene

                # event = dvs_stack.pop()
                # timestamp = time_stack.pop()

                image = image_queue.get()
                frame_path = '%06d' % image.frame
                image.save_to_disk(output_path + image_path + frame_path + '.png') # YOLO format
                # cv2.imwrite(output_path + event_path + frame_path + '.png', event)
                with open(output_path + rgb_label_path + frame_path+".txt", "w", encoding = "utf-8") as file:
                    for bbox in bboxes:
                        file.write(bbox[0]+f" {bbox[1]} {bbox[2]} {bbox[3]} {bbox[4]}\n")
                    file.close()
                if time_queue.empty() != True:
                    events_list = time_queue.get()
                    dvs_events = np.frombuffer(events_list.raw_data, dtype=np.dtype([('x', np.uint16), ('y',np.uint16), ('t',np.int64), ('pol', np.bool)]))
                    dvs_event_copy = dvs_events.copy()
                    dvs_event_copy['t'] //= 1000
                    with open(dvs_output_path, mode="a",  newline='') as file:
                        writer = csv.writer(file)
                        for event in dvs_event_copy:
                            writer.writerow(event)
                        file.close()
                    timestamp = dvs_event_copy[0]['t']
                    with open("output/bbox.csv", "a", encoding = "utf-8") as file:
                        writer = csv.writer(file)
                        for bbox in bboxes_dvs:
                            event = [timestamp] + bbox
                            writer.writerow(event)
                        file.close()
                else:
                    print("time queue is empty, current frame:", image.frame)

                if times % 100 == 0:
                    print("times = ", times)
                if times > 1000:
                    camera.stop()
                    raw_camera.stop()
                    print("Finish")
                    break
            index += 1
            if times > 1000:
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
