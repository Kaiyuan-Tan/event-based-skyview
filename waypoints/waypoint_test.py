import carla
import time
import random

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10000.0)
    world = client.get_world()
    map = world.get_map()
    spectator = world.get_spectator()
    blueprint_library = world.get_blueprint_library()

    for actor in world.get_actors().filter('vehicle.*'):
        actor.destroy()

    vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]

    start_location = carla.Location(x=-47, y=-62)
    start_waypoint = map.get_waypoint(start_location, project_to_road=True)
    current_wp = start_waypoint
    # for _ in range(500):
    #     next_wps = current_wp.next(10.0)
    #     if not next_wps:
    #         break
    #     current_wp = next_wps[0]
    #     route.append(current_wp)

    # for wp in route:
    #     world.debug.draw_point(wp.transform.location, size=0.2, color=carla.Color(0,255,0), life_time=20.0)

    # for wp in route:
    #     spectator.set_transform(wp.transform)
    #     time.sleep(0.1)

    while True:
        spectator.set_transform(current_wp.transform)
        time.sleep(0.5)
        next_wps = current_wp.next(0.1)
        current_wp = next_wps[0]

    print("完成路径导航")
    time.sleep(3)
    vehicle.destroy()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n退出。')
