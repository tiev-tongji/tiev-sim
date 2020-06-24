#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

"""
Welcome to CARLA manual control.

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    AD           : steer
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot
    L            : toggle HIL mode
    M            : toggle manual transmission
    ,/.          : gear up/down

    TAB          : change sensor position
    `            : next sensor
    [1-9]        : change to sensor [1-9]
    C            : change weather (Shift+C reverse)
    Backspace    : change vehicle

    R            : toggle recording images to disk

    CTRL + R     : toggle recording of simulation (replacing any previous)
    CTRL + P     : start replaying last recorded simulation
    CTRL + +     : increments the start time of the replay by 1 second (+SHIFT = 10 seconds)
    CTRL + -     : decrements the start time of the replay by 1 second (+SHIFT = 10 seconds)

    F1           : toggle HUD
    H/?          : toggle help
    ESC          : quit
"""

from __future__ import print_function


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla

from carla import ColorConverter as cc

import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref
import threading
import multiprocessing
from multiprocessing import Process
from multiprocessing import Manager 
import time
#from compiler.ast import flatten
import csv

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_b
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_m
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
    from pygame.locals import K_l 
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    import pandas as pd
except ImportError:
    raise RuntimeError('cannot import pandas, make sure numpy package is installed')

try:
    import lcm as xcm
    print("use lcm")
    xcm_version = "LCM"
except ImportError:
    try:
        import zerocm as xcm
        print("use zcm")
        xcm_version = "ZCM"
    except ImportError:
        raise RuntimeError('cannot import lcm or zcm, make sure lcm package is installed')

sys.path.append('../')
if xcm_version=="LCM":
    try:
        import icumsg_lcm
        from icumsg_lcm import structNAVINFO
        from icumsg_lcm import structFUSIONMAP
        #from icumsg_lcm import structCARCONTROL
        #from icumsg_lcm import structAIMPATHINT
        from icumsg_lcm import structCANCONTROL
        from icumsg_lcm import structCANINFO
        from icumsg_lcm import LinePoint, LANE, structLANES
    except ImportError:
        raise RuntimeError("cannot import lcm defination")
elif xcm_version=="ZCM":
    try:
        import icumsg_zcm
        from icumsg_zcm import structNAVINFO
        from icumsg_zcm import structFUSIONMAP
        #from icumsg_zcm import structCARCONTROL
        #from icumsg_zcm import structAIMPATHINT
        from icumsg_zcm import structCANINFO
        from icumsg_zcm import structCANCONTROL
        from icumsg_zcm import LinePoint, LaneLine, LANE, structLANES
        from icumsg_zcm import POSITION, BOUNDINGBOX, OBJECT, structOBJECTLIST
        from icumsg_zcm import pt, structCARLACLOUD
    except ImportError:
        raise RuntimeError("cannot import zcm defination")

try:
    import utm
except ImportError:
    raise RuntimeError('cannot import utm, make sure utm package is installed')


# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


if xcm_version=="LCM":
    tunnel = xcm.LCM()
elif xcm_version=="ZCM":
    tunnel = xcm.ZCM("udpm://239.255.76.67:7667?ttl=1")

navinfo = structNAVINFO()
fusionmap = structFUSIONMAP()
fusionmap.resolution = 0.2
fusionmap.cols = 151
fusionmap.rows = 401
fusionmap.center_col = 75
fusionmap.center_row = 300
#carcontrol = structCARCONTROL()
#aimpathint = structAIMPATHINT()
#cancontrol = structCANCONTROL()
caninfo = structCANINFO()
lanes = structLANES()

vehicle_steer = 0.0
vehicle_speed = 0.0

aim_speed = 0.0
aim_steer = 0.0

navinfo_csv = open("navinfo.csv", 'a')
navinfo_writer = csv.writer(navinfo_csv, dialect='excel')
navinfo_writer.writerow(['timestamp', 'lat', 'lon', 'utmX', 'utmY', 'speed', 'heading', 'pitch'])
caninfo_csv = open("caninfo.csv", 'a')
caninfo_writer = csv.writer(caninfo_csv, dialect='excel')
caninfo_writer.writerow(['timestamp', 'carspeed', 'carsteer'])

class CarControlPid():
    def __init__(self, k_p_steer=0.1, k_i_steer = 0.0, k_d_steer = 0.0,
                 k_p_acc=1.0, k_i_acc=0.0, k_d_acc=0.0,
                 k_p_brake=1.0, k_i_brake=0.0, k_d_brake=0.0,
                 dying_zone_acc=0.0, dying_zone_brake=0.0, dying_zone_steer=0.0):
        self.aim_steer = 0.0
        self.aim_speed = 0.0
        self.vehicle_steer = 0.0
        self.vehicle_speed = 0.0
        self.intergral = 0.0
        self.steer_intergral = 0.0
        self.intergral_count = 0
        self.steer_intergral_count = 0
        self.error = 0.0
        self.steer_error = 0.0
        self.steer_last_error = 0.0
        self.last_error = 0.0
        self.throttle = 0.0
        self.brake = 0.0
        self.steer = 0.0
        self.k_p_acc = k_p_acc
        self.k_i_acc = k_i_acc
        self.k_d_acc = k_d_acc
        self.k_p_brake = k_p_brake
        self.k_i_brake = k_i_brake
        self.k_d_brake = k_d_brake
        self.k_p_steer = k_p_steer
        self.k_i_steer = k_i_steer
        self.k_d_steer = k_d_steer
        self.dying_zone_acc = dying_zone_acc
        self.dying_zone_brake = dying_zone_brake
        self.dying_zone_steer = dying_zone_steer

    def calculate_next_control(self):
        # steer: PID
        self.steer_error = self.aim_steer - self.vehicle_steer
        if self.steer_intergral_count == 150:
            self.steer_intergral = 0
            self.steer_intergral_count = 0
        self.steer_intergral += self.steer_error
        self.steer_intergral_count += 1
        if abs(self.steer_error)>self.dying_zone_steer:
            #self.steer = self.k_p_steer * self.steer_error \
            #             + self.k_d_steer * (self.steer_error - self.steer_last_error) + \
            #             self.k_i_steer * self.steer_intergral
            self.steer = self.k_p_steer * self.aim_steer
            #print("aimsteer: %s, applysteer: %s" % (self.aim_steer, self.steer))
        else:
            self.steer = self.vehicle_steer
        # acc/brake: PID 
        self.error = self.aim_speed - self.vehicle_speed
        #print("error: %s, last error: %s" % (self.error, self.steer_last_error))
        if self.intergral_count == 150:
            self.intergral = 0
            self.intergral_count = 0
        self.intergral += self.error
        self.intergral_count += 1
        if self.aim_speed==0:
            self.brake = 1.0
            self.throttle = 0.0
        elif self.error > self.dying_zone_acc:
            self.brake = 0.0
            self.throttle = \
                self.k_p_acc * self.error + \
                self.k_i_acc * self.intergral + \
                self.k_d_acc * (self.error - self.last_error)
        elif self.error < -self.dying_zone_brake:
            self.throttle = 0.0
            self.brake = \
                self.k_p_brake * abs(self.error) + \
                self.k_i_brake * abs(self.intergral) + \
                self.k_d_acc * (abs(self.error) - abs(self.last_error))
        else:
            self.throttle = 0.0
            self.brake = 0.0
        self.last_error = self.error
        self.steer_last_error = self.steer_error

    def limit_control(self):
        if self.steer>1:
            self.steer = 1
        elif self.steer<-1:
            self.steer = -1
        else:
            pass
        if self.throttle>1:
            self.throttle = 1
        elif self.throttle<0:
            self.throttle = 0
        if self.brake>1:
            self.brake = 1
        elif self.brake<0:
            self.brake = 0

#manager = Manager()


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


def car_control_handler(channel, data): # old
    msg = structCARCONTROL.decode(data)
    global xcm_acc, xcm_brake, xcm_steer
    xcm_acc = msg.acc
    xcm_brake = msg.brake
    xcm_steer = msg.steer
    #print('received carcontrol', xcm_acc, xcm_brake, xcm_steer)

def aimpathint_handler(channel, data): # old
    msg = structAIMPATHINT.decode(data)
    #global aimpathint
    pass #TODO

def can_control_handler(channel, data):
    msg = structCANCONTROL.decode(data)
    global xcm_acc, xcm_brake, xcm_steer, aim_speed, aim_steer
    aim_steer = msg.aimsteer # deg
    aim_speed = msg.aimspeed # kph
    #print("recieved CAN CONTROL: (%f.kph, %f.deg)" % (aim_speed, aim_steer))

def get_xcm():
    global tunnel
    while True:
        try:
            tunnel.handle()
        except KeyboardInterrupt:
            pass

def start_sub():
    global tunnel, can_control_handler
    if xcm_version=="LCM":
        #sub_carcontrol = tunnel.subscribe('CARCONTROL', car_control_handler)
        #sub_aimpathint = tunnel.subscribe('AIMPATHINT', aimpathint_handler)
        sub_cancontrol = tunnel.subscribe('CANCONTROL', can_control_handler)
    elif xcm_version=="ZCM":
        #sub_carcontrol = tunnel.subscribe_raw('CARCONTROL', car_control_handler)
        #sub_aimpathint = tunnel.subscribe_raw('AIMPATHINT', aimpathint_handler)
        sub_cancontrol = tunnel.subscribe_raw('CANCONTROL', can_control_handler)
    if xcm_version=="LCM":
        thread_sub = threading.Thread(target=get_xcm, args=())
        thread_sub.start()
    elif xcm_version=="ZCM":
        tunnel.start()

# Unreal to vehicle frame reference system
def Unreal2VF(theta_rad, x, y):
    yawMatrix = np.matrix([ \
        [math.cos(theta_rad), math.sin(theta_rad)], \
        [-math.sin(theta_rad), math.cos(theta_rad)] \
    ])
    return np.linalg.inv(yawMatrix) \
        .dot(np.array([[0,1],[1,0]])) \
        .dot(np.array([[x], [y]]))


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


class World(object):
    def __init__(self, carla_world, hud, args):
        self.world = carla_world
        self.actor_role_name = args.rolename
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None
        self.lane_detector = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._gamma = args.gamma
        self.restart()
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0

    def restart(self):
        settings = self.world.get_settings()
        #settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # Get a random blueprint.
        blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))  # random choose a car model.
        #blueprint = random.choice(self.world.get_blueprint_library().filter('vehicle.carsim.*'))   # only use carsim car as player, only used in Carsim/Carla env.
        blueprint.set_attribute('role_name', self.actor_role_name)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'true')
        # Spawn the player.
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        while self.player is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)
            spawn_points = self.map.get_spawn_points()
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            #31.23152655951361, 7.47199273909087
            origin_lat = 31.22974718964157
            origin_lon = 121.47423179127021
            spawn_point.location.y = utm.from_latlon(origin_lat, origin_lon)[0] - 354664.1253431414
            spawn_point.location.x = utm.from_latlon(origin_lat, origin_lon)[1] - 3456156.075134088
            print(spawn_point.location.x, spawn_point.location.y)
            spawn_point.rotation.yaw = 90
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma, self.gnss_sensor)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        self.lane_detector = LaneDetector(self.player)
        #self.camera_manager.another_lidar() # add by leoherz
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def tick(self, clock):
        self.hud.tick(self, clock)
        #self.lane_detector.print_waypoint()
        #self.lane_detector.generate_waypoints([i*0.2 for i in range(1,100)], self.map)
        #print([i*0.2 for i in range(1,100)])
        #self.lane_detector.generate_waypoints([0.5,1,1.5,2,2.5], self.map)
        #self.lane_detector.generate_waypoints([1], self.map)
        #self.lane_detector.calculate_line_points()
        #self.lane_detector.pack_lanes(1)

    def get_waypoints(self, distance):
        return self.current_waypoint.next(distance)[0]

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        global caninfo_csv, navinfo_csv
        actors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.player,
            self.camera_manager.lidar]
        for actor in actors:
            if actor is not None:
                actor.destroy()
        self.gnss_sensor.roadmap.close()
        caninfo_csv.close()
        navinfo_csv.close()

    def pub_objectlist(self):
        global tunnel, xcm_version
        actors = self.world.get_actors()
        # for actor in actors:
        #    print(actor)
        objectlist = structOBJECTLIST()
        objectlist.timestamp = 0
        objectlist.data_source = 1

        for index, actor in enumerate(actors):
            # print(actor)
            obj = OBJECT()
            obj.id = index
            if actor.type_id.startswith('vehicle'):
                if actor.attributes['number_of_wheels'] == 4:
                    obj.obj_type = 0
                elif actor.attributes['number_of_wheels'] == 2:
                    obj.obj_type = 1
            elif actor.type_id.startswith('walker'):
                obj.obj_type = 2
            else:
                continue
            v = actor.get_velocity()
            obj.v = np.linalg.norm((v.x, v.y, v.z))
            # Unreal: xyz, and front is x+
            # vehicle frame: xy, and front is y+
            # -180~180
            theta1 = self.player.get_transform().rotation.yaw
            theta2 = actor.get_transform().rotation.yaw
            # TODO: 以下暂时只考虑水平面
            theta = theta2 - theta1 + 90
            # -pi~pi
            theta = theta + 360 if theta < -180 else theta
            theta = theta - 360 if theta > 180 else theta
            obj.theta = theta / 180 * math.pi
            # self location
            loc0 = self.player.get_location()
            # actor location
            loc1 = actor.get_location()
            # relative location
            rloc = loc1 - loc0
            # Unreal to vehicle frame
            loc = POSITION()
            loc.x = rloc.y
            loc.y = rloc.x
            # 8 vertices of bounding box in relative location
            vertices = []
            bb = actor.bounding_box
            for i in range(2):
                for j in range(2):
                    vertices.append((loc1.y + bb.location.y + (bb.extent.y if i > 0 else -bb.extent.y) - loc0.y,
                                     loc1.x + bb.location.x + (bb.extent.x if j > 0 else -bb.extent.x) - loc0.x))
            xmax = sys.float_info.min
            xmin = sys.float_info.max
            ymax = sys.float_info.min
            ymin = sys.float_info.max
            for vertex in vertices:
                xmax = vertex[0] if vertex[0] > xmax else xmax
                xmin = vertex[0] if vertex[0] < xmin else xmin
                ymax = vertex[1] if vertex[1] > ymax else ymax
                ymin = vertex[1] if vertex[1] < ymin else ymin
            obj.width = xmax - xmin
            obj.length = ymax - ymin
            tmp = POSITION()
            tmp.x = xmax; tmp.y = ymax
            obj.corners.p1 = tmp
            tmp.x = xmax; tmp.y = ymin
            obj.corners.p2 = tmp
            tmp.x = xmin; tmp.y = ymin
            obj.corners.p3 = tmp
            tmp.x = xmin; tmp.y = ymax
            obj.corners.p4 = tmp
            # relative speed
            # TODO: 以下暂时只考虑水平面
            relv_scalar = np.linalg.norm((v.y, v.x))
            relv = (relv_scalar * np.cos(theta), relv_scalar * np.sin(theta))
            pred = POSITION()
            pred.x = loc.x
            pred.y = loc.y
            for i in range(0, 5):
                obj.path.append(pred)
                pred.x = pred.x + relv[0]
                pred.y = pred.y + relv[1]
            obj.pathNum = len(obj.path)
            objectlist.obj.append(obj)
        objectlist.count = len(objectlist.obj)

        try:
            if xcm_version == "LCM":
                tunnel.publish('OBJECTLIST', objectlist.encode())
            elif xcm_version == "ZCM":
                tunnel.publish('OBJECTLIST', objectlist)
        except:
            print('error publishing OBJECTLIST!')


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    def __init__(self, world, start_in_autopilot):
        self.world = world
        self.pid_controller = CarControlPid()
        self._hilpilot_enabled = False #add a flag presenting hil mode
        self._autopilot_enabled = start_in_autopilot
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            world.player.set_autopilot(self._autopilot_enabled)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._autopilot_enabled = False
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self, client, world, clock):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_BACKSPACE:
                    world.restart()
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    world.hud.help.toggle()
                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                elif event.key > K_0 and event.key <= K_9:
                    world.camera_manager.set_sensor(event.key - 1 - K_0)
                elif event.key == K_r and not (pygame.key.get_mods() & KMOD_CTRL):
                    world.camera_manager.toggle_recording()
                elif event.key == K_r and (pygame.key.get_mods() & KMOD_CTRL):
                    if (world.recording_enabled):
                        client.stop_recorder()
                        world.recording_enabled = False
                        world.hud.notification("Recorder is OFF")
                    else:
                        client.start_recorder("manual_recording.rec")
                        world.recording_enabled = True
                        world.hud.notification("Recorder is ON")
                elif event.key == K_p and (pygame.key.get_mods() & KMOD_CTRL):
                    # stop recorder
                    client.stop_recorder()
                    world.recording_enabled = False
                    # work around to fix camera at start of replaying
                    currentIndex = world.camera_manager.index
                    world.destroy_sensors()
                    # disable autopilot
                    self._autopilot_enabled = False
                    world.player.set_autopilot(self._autopilot_enabled)
                    world.hud.notification("Replaying file 'manual_recording.rec'")
                    # replayer
                    client.replay_file("manual_recording.rec", world.recording_start, 0, 0)
                    world.camera_manager.set_sensor(currentIndex)
                elif event.key == K_MINUS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start -= 10
                    else:
                        world.recording_start -= 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                elif event.key == K_EQUALS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start += 10
                    else:
                        world.recording_start += 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                elif event.key == K_b and not (pygame.key.get_mods() & KMOD_CTRL):
                    self.world.gnss_sensor.roadmap_stat = not self.world.gnss_sensor.roadmap_stat
                    if self.world.gnss_sensor.roadmap_stat==True:
                        world.hud.notification("start roadmap recorder")
                    else:
                        world.hud.notification("stop roadmap recorder")
                if isinstance(self._control, carla.VehicleControl):
                    if event.key == K_q:
                        self._control.gear = 1 if self._control.reverse else -1
                    elif event.key == K_m:
                        self._control.manual_gear_shift = not self._control.manual_gear_shift
                        self._control.gear = world.player.get_control().gear
                        world.hud.notification('%s Transmission' %
                                               ('Manual' if self._control.manual_gear_shift else 'Automatic'))
                    elif self._control.manual_gear_shift and event.key == K_COMMA:
                        self._control.gear = max(-1, self._control.gear - 1)
                    elif self._control.manual_gear_shift and event.key == K_PERIOD:
                        self._control.gear = self._control.gear + 1
                    elif event.key == K_p and not (pygame.key.get_mods() & KMOD_CTRL):
                        if self._hilpilot_enabled & (not self._autopilot_enabled):
                            # if hil mode running, turn off it
                            self._hilpilot_enabled = False
                            world.hud.notification('Hil Mode %s' % ('On' if self._hilpilot_enabled else 'Off'))
                        self._autopilot_enabled = not self._autopilot_enabled
                        world.player.set_autopilot(self._autopilot_enabled)
                        world.hud.notification('Carla built-in Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))
                    elif event.key == K_l and not (pygame.key.get_mods() & KMOD_CTRL):
                        if self._autopilot_enabled & (not self._hilpilot_enabled):
                            # if built-in autopilot running, turn off it
                            self._autopilot_enabled = False
                            world.player.set_autopilot(self._autopilot_enabled)
                            world.hud.notification('Carla built-in Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))
                        self._hilpilot_enabled = not self._hilpilot_enabled
                        world.hud.notification('Hil Mode %s' % ('On' if self._hilpilot_enabled else 'Off'))
        if (not self._autopilot_enabled)&(not self._hilpilot_enabled):
            if isinstance(self._control, carla.VehicleControl):
                self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
                self._control.reverse = self._control.gear < 0
            elif isinstance(self._control, carla.WalkerControl):
                self._parse_walker_keys(pygame.key.get_pressed(), clock.get_time())
            world.player.apply_control(self._control)
        elif self._hilpilot_enabled:# hil mode run
            if isinstance(self._control, carla.VehicleControl):
                self._parse_xcm_control()# parse xcm control command from icu
                world.player.apply_control(self._control)

    def _parse_xcm_control(self):
        global aim_speed, aim_steer, vehicle_steer, vehicle_speed
        self.pid_controller.aim_speed = aim_speed / 3.6
        self.pid_controller.aim_steer = aim_steer / 540.0
        self.pid_controller.vehicle_speed = vehicle_speed / 360.0
        self.pid_controller.vehicle_steer = vehicle_steer / 540.0
        # pid parameters
        self.pid_controller.k_p_steer = 1
        #self.pid_controller.k_i_steer = 0.2
        #self.pid_controller.k_d_steer = 0.05
        self.pid_controller.k_p_acc = 0.1
        self.pid_controller.k_i_acc = 0.005
        self.pid_controller.k_d_acc = 0
        self.pid_controller.k_p_brake = 0.2
        self.pid_controller.k_i_brake = 0
        self.pid_controller.k_d_brake = 0.1
        self.pid_controller.dying_zone_steer = 0.004 # 1.08deg
        self.pid_controller.dying_zone_acc = 0.01 # 3m/s.delta_t ~ 0.75
        self.pid_controller.dying_zone_brake = 4 # -3m/s.delta_t ~ 0.75 delta_t about 0.05s
        # calculate next control
        self.pid_controller.calculate_next_control()
        self.pid_controller.limit_control()
        # apply control
        self._control.steer = self.pid_controller.steer
        self._control.throttle = self.pid_controller.throttle
        self._control.brake = self.pid_controller.brake
        self._control.hand_brake = 0

    def _parse_vehicle_keys(self, keys, milliseconds):
        self._control.throttle = 1.0 if keys[K_UP] or keys[K_w] else 0.0
        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            if self._steer_cache > 0:
                self._steer_cache = 0
            else:
                self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            if self._steer_cache < 0:
                self._steer_cache = 0
            else:
                self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
        self._control.hand_brake = keys[K_SPACE]

    def _parse_walker_keys(self, keys, milliseconds):
        self._control.speed = 0.0
        if keys[K_DOWN] or keys[K_s]:
            self._control.speed = 0.0
        if keys[K_LEFT] or keys[K_a]:
            self._control.speed = .01
            self._rotation.yaw -= 0.08 * milliseconds
        if keys[K_RIGHT] or keys[K_d]:
            self._control.speed = .01
            self._rotation.yaw += 0.08 * milliseconds
        if keys[K_UP] or keys[K_w]:
            self._control.speed = 3.333 if pygame.key.get_mods() & KMOD_SHIFT else 1.389
        self._control.jump = keys[K_SPACE]
        self._rotation.yaw = round(self._rotation.yaw, 1)
        self._control.direction = self._rotation.get_forward_vector()

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        fonts = [x for x in pygame.font.get_fonts() if 'mono' in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()
        self.pid_controller = None

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        global aim_steer, vehicle_steer
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        try:
            max_wheel_angle = world.player.get_physics_control().wheels[0].max_steer_angle / 2.0
            wheel_friction = world.player.get_physics_control().wheels[0].tire_friction
        except:
            max_wheel_angle = 45
            wheel_friction = 0.1
        #waypoint = world.map.get_waypoint(t.location, project_to_road=True, lane_type=(
                #carla.LaneType.Driving | carla.LaneType.Shoulder | carla.LaneType.Sidewalk))
        #lane_change = waypoint.lane_change
        #left_line = waypoint.left_lane_marking
        #right_line = waypoint.right_lane_marking
        heading = 'N' if abs(t.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(t.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > t.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > t.rotation.yaw > -179.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name,
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (t.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            #'Height:  % 18.0f m' % t.location.z,
            #'wheel steer angle: %f' % float(c.steer*max_wheel_angle),
            #'max wheel angle: %f' % max_wheel_angle,
            #'tire friction: %f' % wheel_friction,
            #'lane change: %s' % lane_change,
            #'left line marking: %s' % left_line.type,
            #'right line marking: %s' % right_line.type,
            '',
            #'speed error: %s' % self.pid_controller.error,
            #'steer error: %s' % self.pid_controller.steer_error,
            #'aim steer: %s' % int(aim_steer),
            #'vehicle steer: %s' % int(vehicle_steer),
            '']
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
            for d, vehicle in sorted(vehicles):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))


# ==============================================================================
# -- GnssSensor --------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    def __init__(self, parent_actor):
        self.roadmap = open("roadmap.txt",'a')
        self.roadmap.write("Id Lon Lat heading curvature mode SpeedMode EventMode"
                           " OppositeSideMode LaneNum LaneSeq LaneWidth\n")
        self.roadmap_id = 0
        self.roadmap_stat = False

        self.carPos = np.eye(4, dtype = float)
        self.originPos = utm.to_latlon(354664.1253431414, 3456156.075134088, 51, 'R')
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        global navinfo, xcm_version, tunnel, caninfo, vehicle_steer, vehicle_speed
        self = weak_self()
        if not self:
            return
        #self.lat = event.latitude
        #self.lon = event.longitude

        # pack navinfo
        t = self._parent.get_transform()
        v = self._parent.get_velocity()
        steer = self._parent.get_control().steer
        try:
            max_wheel_angle = self._parent.get_physics_control().wheels[0].max_steer_angle / 2.0 # double side max angle together
        except:
            max_wheel_angle = 45
        translation = (t.location.x, t.location.y, t.location.z)
        #print("UE4 translation (%f, %f, %f)"%translation)
        yaw = t.rotation.yaw
        roll = t.rotation.roll
        pitch = t.rotation.pitch
        yawMatrix = np.matrix([ \
                [math.cos(yaw), -math.sin(yaw), 0], \
                [math.sin(yaw), math.cos(yaw), 0], \
                [0, 0, 1] \
        ])
        pitchMatrix = np.matrix([ \
                [math.cos(pitch), 0, math.sin(pitch)], \
                [0, 1, 0], \
                [-math.sin(pitch), 0, math.cos(pitch)] \
        ])
        rollMatrix = np.matrix([ \
                [1, 0, 0], \
                [0, math.cos(roll), -math.sin(roll)], \
                [0, math.sin(roll), math.cos(roll)] \
        ])
        R = yawMatrix.dot(pitchMatrix).dot(rollMatrix)
        self.carPos[:3,:3] = R
        self.carPos[:3,3] = translation
        yawtemp = yaw - 90.0
        if(yawtemp < -179.999):
            yawtemp += 360.0
        
        nowUtm = [354664.1253431414 + translation[1], 3456156.075134088 + translation[0]]
        gps = utm.to_latlon(nowUtm[0], nowUtm[1], 51, 'R')
        navinfo.timestamp = 0            # not be used, default 0
        navinfo.mLat = gps[0]            # gps, utm calculate from Translation, lijun
        navinfo.mLon = gps[1]
        #print('lat lon', gps)
        navinfo.utmX = nowUtm[0]
        navinfo.utmY = nowUtm[1]
        navinfo.mAlt = translation[2]
        navinfo.mHeading = math.radians(-yawtemp)
        #print('heading ', navinfo.mHeading)
        navinfo.mPitch = pitch           # not be used
        navinfo.mAngularRateZ = 0.0      # not be used, default 0
        navinfo.mSpeed3d = float(math.sqrt(v.x**2 + v.y**2 + v.z**2))
        navinfo.mVe = v.x                # not be used, default 0
        navinfo.mVn = v.y                # not be used, default 0
        navinfo.mCurvature = 0.0
        navinfo.mRTKStatus = 1           # not be used, default 1
        navinfo.mHPOSAccuracy = 0.01     # not be used, default 0.01
        navinfo.isReckoningVaild = 0     # not be used, default 0
        navinfo.mGpsNumObs = 11          # not be used, default 11
        navinfo.mAx = 0.0                # not be used, default 0
        navinfo.mAy = 0.0                # not be used, default 0
        caninfo.timestamp = 0
        caninfo.carspeed = int(navinfo.mSpeed3d*3.6*100)
        #caninfo.carsteer = int(steer*max_wheel_angle)
        caninfo.carsteer = int(steer*-540.0)
        vehicle_steer = caninfo.carsteer
        vehicle_speed = caninfo.carspeed
        #print("CANINFO: " ,caninfo.timestamp, caninfo.carspeed, caninfo.carsteer)
        self.lat = navinfo.mLat
        self.lon = navinfo.mLon
        # record roadmap
        if self.roadmap_stat==True:
            to_write = str(self.roadmap_id)+' '+str(self.lat)+' '+str(self.lon)+\
                ' '+str(navinfo.mHeading)+' '+str(navinfo.mCurvature)+' '+'11 3 0 3 0 0 0.00000000000000\n'
            self.roadmap.write(to_write)
            self.roadmap_id += 1
        # sync publish navinfo， caninfo
        '''
        if xcm_version=="LCM":
            tunnel.publish("NAVINFO", navinfo.encode())
        elif xcm_version=="ZCM":
            tunnel.publish("NAVINFO", navinfo)
        if xcm_version=="LCM":
            tunnel.publish("CANINFO", caninfo.encode())
        elif xcm_version=="ZCM":
            tunnel.publish("CANINFO", caninfo)
        '''

    @staticmethod
    def start_pub_navinfo(freq=0.0):
        # async publish
        pub = threading.Thread(target=GnssSensor.pub_navinfo, args=(freq,))
        pub.start()

    @staticmethod
    def start_pub_caninfo(freq=0.0):
        # async publish
        pub = threading.Thread(target=GnssSensor.pub_caninfo, args=(freq,))
        pub.start()

    @staticmethod
    def pub_navinfo(freq):
        global navinfo, tunnel, xcm_version, navinfo_writer
        if xcm_version=="LCM":
            try:
                while True:
                    tunnel.publish('NAVINFO', navinfo.encode())
                    #print('publish NAVINFO!')
                    time.sleep(1/freq)
            except:
                pass
        elif xcm_version=="ZCM":
            try:
                while True:
                    tunnel.publish("NAVINFO", navinfo)
                    navinfo_writer.writerow([time.time(), navinfo.mLat, navinfo.mLon, navinfo.utmX,
                                             navinfo.utmY, navinfo.mSpeed3d, navinfo.mHeading, navinfo.mPitch])
                    #print('publish NAVINFO!')
                    time.sleep(1/freq)
            except:
                pass

    @staticmethod
    def pub_caninfo(freq):
        global caninfo, tunnel, xcm_version, caninfo_writer
        if xcm_version=="LCM":
            try:
                while True:
                    tunnel.publish('CANINFO', caninfo.encode())
                    #print('publish CANINFO!')
                    time.sleep(1/freq)
            except:
                pass
        elif xcm_version=="ZCM":
            try:
                while True:
                    tunnel.publish("CANINFO", caninfo)
                    caninfo_writer.writerow([time.time(), caninfo.carspeed, caninfo.carsteer])
                    #print('publish CANINFO!')
                    time.sleep(1/freq)
            except:
                pass


# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud, gamma_correction, gnss_sensor):
        self.gnss_sensor = gnss_sensor
        self.lidarData = np.zeros([1,3])
        self.lidar = None
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        Attachment = carla.AttachmentType
        self._camera_transforms = [
            (carla.Transform(carla.Location(x=0.0, y=0.0, z=2.2), carla.Rotation(pitch=0, roll=0.0, yaw=0.0)), Attachment.Rigid),
            (carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=1.6, z=1.7)), Attachment.Rigid),
            (carla.Transform(carla.Location(x=5.5, y=1.5, z=1.5)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=-1, y=-bound_y, z=0.5)), Attachment.Rigid)]
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)', {}],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)', {}],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)', {}],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)', {}],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
                'Camera Semantic Segmentation (CityScapes Palette)', {}],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)', {}],
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB Distorted',{}]]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
                if bp.has_attribute('gamma'):
                    bp.set_attribute('gamma', str(gamma_correction))
                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
            elif item[0].startswith('sensor.lidar'):
                self.lidar_range = 50 # meters (0.9.6 or previous: centimeters)
                bp.set_attribute('range', str(self.lidar_range))
                bp.set_attribute('rotation_frequency', str(40))#20
                bp.set_attribute('channels', str(64))
                bp.set_attribute('points_per_second', str(350000))#450000
                bp.set_attribute('upper_fov', str(10))
                bp.set_attribute('lower_fov', str(-30))
                print(bp.get_attribute('channels'))
                print(bp.get_attribute('range'))
                print(bp.get_attribute('points_per_second'))
                print(bp.get_attribute('rotation_frequency'))
                print(bp.get_attribute('upper_fov'))
                print(bp.get_attribute('lower_fov'))
            item.append(bp)

        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else \
            (force_respawn or (self.sensors[index][2] != self.sensors[self.index][2]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    # make another lidar always be running in simulation
    def another_lidar(self):
        index = 7
        transform_index = self.transform_index # TODO: lidar position adjustment
        self.lidar = self._parent.get_world().spawn_actor(
            self.sensors[index][-1],
            self._camera_transforms[transform_index][0],
            attach_to=self._parent,
            attachment_type=self._camera_transforms[transform_index][1])
        weak_self = weakref.ref(self)
        self.lidar.listen(lambda image: CameraManager._parse_lidar(weak_self, image))
        self.hud.notification("lidar is running: backgroup mode")

    @staticmethod
    def _parse_lidar(weak_self, image):
        global xcm_version, tunnel, fusionmap
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            start = time.time()
            #print("frame id: %d"%(image.frame_number))
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0]/3), 3))
            dx = image.transform.location.x
            dy = image.transform.location.y
            dz = image.transform.location.z
            roll = image.transform.rotation.roll
            pitch = image.transform.rotation.pitch
            yaw = image.transform.rotation.yaw
            yawMatrix = np.matrix([ \
                    [math.cos(yaw), -math.sin(yaw), 0], \
                    [math.sin(yaw), math.cos(yaw), 0], \
                    [0, 0, 1] \
            ])
            pitchMatrix = np.matrix([ \
                    [math.cos(pitch), 0, math.sin(pitch)], \
                    [0, 1, 0], \
                    [-math.sin(pitch), 0, math.cos(pitch)] \
            ])
            rollMatrix = np.matrix([ \
                    [1, 0, 0], \
                    [0, math.cos(roll), -math.sin(roll)], \
                    [0, math.sin(roll), math.cos(roll)] \
            ])
            Rtmp = yawMatrix.dot(pitchMatrix).dot(rollMatrix) #zyx
            lidarT = np.eye(4, dtype = float)
            lidarT[:3,:3] = Rtmp
            lidarT[:3,3] = (dx, dy, dz)
            #print("lidarT")
            #print(lidarT.astype(np.dtype('f4')))
            
            carT = self.gnss_sensor.carPos
            #print("carT")
            #print(carT.astype(np.dtype('f4')))
            
            car2lidar = np.linalg.inv(carT).dot(lidarT)
            
            #print("car2lidarT")
            #print(car2lidar.astype(np.dtype('f4')))
            #points = ((yawMatrix.dot(pointstest.T)).T).astype(np.dtype('f4'))
            
            self.lidarData = np.vstack((self.lidarData,points))
            #print("receiver num: %s" %points.shape[0])
            if(self.lidarData.shape[0] > 135000):
                frameData = self.lidarData#[:135000,:]
                #print("send frame num: %s" %frameData.shape[0])
                self.lidarData = self.lidarData[135000:,:]
                self.lidarData = np.zeros([1,3], dtype = np.int) #add by lijun
            pointsz = -points[:, 2]
            points = np.c_[points[:,:2], pointsz]     
            topheight = 1 

            pointsTopIndex = (points[:,2] < topheight)
            points = points[np.where(pointsTopIndex == True)]
            tempData = current_data.copy()
            current_data = np.array(points[:, :2])
            tempData *= 5
            tempData += (0.5 * 151, 300)
            tempData = tempData.astype(np.int32)
            tempData = np.reshape(tempData, (-1, 2))
            temp_img_size = (401 , 151)
            temp_img = np.zeros(temp_img_size, dtype = np.int32)
            heightdiff = 0.1
            indices = ((tempData[:,0] >= 0) & (tempData[:,0] < 151) & (tempData[:,1] >= 0) & (tempData[:,1] < 401))
            tempData = tempData[np.where(indices == True)]
            pointstmp = points[np.where(indices == True)]
            pointstmp[:,:2] = tempData[:,:2]
            df = pd.DataFrame({'row' : pointstmp[:,1].astype(np.int32), 'col' : pointstmp[:,0].astype(np.int32), 'z' : pointstmp[:,2]})
            groupmin = -df.groupby(['row', 'col']).max().reset_index().values
            groupmax = -df.groupby(['row', 'col']).min().reset_index().values
            groupdata = groupmin.copy()
            groupdata[:,2] = groupmax[:,2] - groupmin[:,2]
            indexz = groupdata[:,2] > heightdiff
            groupdata = groupdata[np.where(indexz == True)][:,:2].astype(np.int32)
            #print ("groupdata:", groupdata, ":groupdata")
            temp_img[tuple(groupdata.T)] = (2)#old 1
            #print ("tmp_image:", temp_img, ":tmp_image")
            lasermsg = structFUSIONMAP()
            lasermsg.resolution = 0.2
            lasermsg.cols = 151
            lasermsg.rows = 401
            lasermsg.center_col = 75
            lasermsg.center_row = 300
            lasermsg.cells = temp_img.tolist()
            if xcm_version == "LCM":
                tunnel.publish("FUSIONMAP",lasermsg.encode())
            elif xcm_version == "ZCM":
                tunnel.publish("FUSIONMAP",lasermsg)
            end = time.time()
            #print("FUSIONMAP TIME:" ,(start-end))
        

    @staticmethod
    def _parse_image(weak_self, image):
        global xcm_version, tunnel, fusionmap
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            start = time.time()
            #print("frame id: %d"%(image.frame_number))
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0]/3), 3))
            dx = image.transform.location.x
            dy = image.transform.location.y
            dz = image.transform.location.z
            roll = image.transform.rotation.roll
            pitch = image.transform.rotation.pitch
            yaw = image.transform.rotation.yaw
            yawMatrix = np.matrix([ \
                    [math.cos(yaw), -math.sin(yaw), 0], \
                    [math.sin(yaw), math.cos(yaw), 0], \
                    [0, 0, 1] \
            ])
            pitchMatrix = np.matrix([ \
                    [math.cos(pitch), 0, math.sin(pitch)], \
                    [0, 1, 0], \
                    [-math.sin(pitch), 0, math.cos(pitch)] \
            ])
            rollMatrix = np.matrix([ \
                    [1, 0, 0], \
                    [0, math.cos(roll), -math.sin(roll)], \
                    [0, math.sin(roll), math.cos(roll)] \
            ])
            Rtmp = yawMatrix.dot(pitchMatrix).dot(rollMatrix) #zyx
            lidarT = np.eye(4, dtype = float)
            lidarT[:3,:3] = Rtmp
            lidarT[:3,3] = (dx, dy, dz)
            #print("lidarT")
            #print(lidarT.astype(np.dtype('f4')))
            
            carT = self.gnss_sensor.carPos
            #print("carT")
            #print(carT.astype(np.dtype('f4')))
            
            car2lidar = np.linalg.inv(carT).dot(lidarT)
            
            #print("car2lidarT")
            #print(car2lidar.astype(np.dtype('f4')))
            #points = ((yawMatrix.dot(pointstest.T)).T).astype(np.dtype('f4'))
            
            self.lidarData = np.vstack((self.lidarData,points))
            #print("receiver num: %s" %points.shape[0])
            if(self.lidarData.shape[0] > 135000):
                frameData = self.lidarData#[:135000,:]
                #print("send frame num: %s" %frameData.shape[0])
                self.lidarData = self.lidarData[135000:,:]
                self.lidarData = np.zeros([1,3], dtype = np.int) #add by lijun
           
            #change unrel z from bottom direction to top direction 
            pointsz = -points[:, 2]
            points = np.c_[points[:,:2], pointsz]     
            #erase point cloud which point.z > 1m
            topheight = 1 
            pointsTopIndex = (points[:,2] < topheight)
            points = points[np.where(pointsTopIndex == True)]

            #carlacloudmsg =  structCARLACLOUD() 
            #carlacloudmsg.num = points.shape[0]
            #for i in range(points.shape[0]):
            #    pc = pt();
            #    pc.x_ = points[i, 0]
            #    pc.y_ = points[i, 1]
            #    pc.z_ = points[i, 2]
            #    carlacloudmsg.data.append(pc)   
            #tunnel.publish("CARLACLOUD",carlacloudmsg)
            #print("publish carla cloud ????????????????")

            current_data = np.array(points[:, :2])
            tempData = current_data.copy()
            tempData *= 5
            tempData += (0.5 * 151, 300)
            tempData = tempData.astype(np.int32)
            tempData = np.reshape(tempData, (-1, 2))
            temp_img_size = (401 , 151)
            temp_img = np.zeros(temp_img_size, dtype = np.int32)
            heightdiff = 0.02
            indices = ((tempData[:,0] >= 0) & (tempData[:,0] < 151) & (tempData[:,1] >= 0) & (tempData[:,1] < 401))
            tempData = tempData[np.where(indices == True)]
            pointstmp = points[np.where(indices == True)]
            pointstmp[:,:2] = tempData[:,:2]
            df = pd.DataFrame({'row' : pointstmp[:,1].astype(np.int32), 'col' : pointstmp[:,0].astype(np.int32), 'z' : pointstmp[:,2]})
            groupmax = df.groupby(['row', 'col']).max().reset_index().values
            groupmin = df.groupby(['row', 'col']).min().reset_index().values
            groupdata = groupmin.copy()
            groupdata[:,2] = groupmax[:,2] - groupmin[:,2]

            indexz = groupdata[:,2] > heightdiff
            
            groupdata = groupdata[np.where(indexz == True)][:,:2].astype(np.int32)
            #print ("groupdata:", groupmin, ":groupdata")
            temp_img[tuple(groupdata.T)] = (2)#old 1
            #print ("tmp_image:", temp_img, ":tmp_image")
            lasermsg = structFUSIONMAP()
            lasermsg.resolution = 0.2
            lasermsg.cols = 151
            lasermsg.rows = 401
            lasermsg.center_col = 75
            lasermsg.center_row = 300
            lasermsg.cells = temp_img.tolist()
            if xcm_version == "LCM":
                tunnel.publish("FUSIONMAP",lasermsg.encode())
            elif xcm_version == "ZCM":
                tunnel.publish("FUSIONMAP",lasermsg)
            end = time.time()
            #print("FUSIONMAP TIME:" ,(start-end))

            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / (2.0 * self.lidar_range)
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)

# ==============================================================================
# -- point_cloud_2_fusion_map --------------------------------------------------
# ==============================================================================

class PCDHandlerProcess(Process):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}):
        self.lidarData = np.zeros([1,3], dtpye=np.float)
        return super(PCDHandlerProcess, self).__init__(group=group, target=target, name=name, args=args, kwargs=kwargs)

    def pcd2fusion_map(self, image, gnss_sensor, tunnel):
        points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 3), 3))
        dx = image.transform.location.x
        dy = image.transform.location.y
        dz = image.transform.location.z
        roll = image.transform.rotation.roll
        pitch = image.transform.rotation.pitch
        yaw = image.transform.rotation.yaw
        yawMatrix = np.matrix([ \
                [math.cos(yaw), -math.sin(yaw), 0], \
                [math.sin(yaw), math.cos(yaw), 0], \
                [0, 0, 1] \
        ])
        pitchMatrix = np.matrix([ \
                [math.cos(pitch), 0, math.sin(pitch)], \
                [0, 1, 0], \
                [-math.sin(pitch), 0, math.cos(pitch)] \
        ])
        rollMatrix = np.matrix([ \
                [1, 0, 0], \
                [0, math.cos(roll), -math.sin(roll)], \
                [0, math.sin(roll), math.cos(roll)] \
        ])
        Rtmp = yawMatrix.dot(pitchMatrix).dot(rollMatrix) #zyx
        lidarT = np.eye(4, dtype = float)
        lidarT[:3,:3] = Rtmp
        lidarT[:3,3] = (dx, dy, dz)
        #print("lidarT")
        #print(lidarT.astype(np.dtype('f4')))
            
        carT = gnss_sensor.carPos
        #print("carT")
        #print(carT.astype(np.dtype('f4')))
            
        car2lidar = np.linalg.inv(carT).dot(lidarT)
            
        #print("car2lidarT")
        #print(car2lidar.astype(np.dtype('f4')))
        #points = ((yawMatrix.dot(pointstest.T)).T).astype(np.dtype('f4'))
            
        self.lidarData = np.vstack((self.lidarData,points))
        print("receiver num: %s" %points.shape[0])
        if(self.lidarData.shape[0] > 135000):
            frameData = self.lidarData#[:135000,:]
            #print("send frame num: %s" %frameData.shape[0])
            self.lidarData = self.lidarData[135000:,:]
            self.lidarData = np.zeros([1,3], dtype = np.int) #add by lijun
        
        pointsz = -points[:, 2]
        points = np.c_[points[:,:2], pointsz]     
        topheight = 1 

        pointsTopIndex = (points[:,2] < topheight)
        points = points[np.where(pointsTopIndex == True)]
        
        current_data = np.array(points[:, :2])
        tempData = current_data.copy()
        tempData *= 5
        tempData += (0.5 * 151, 300)
        tempData = tempData.astype(np.int32)
        tempData = np.reshape(tempData, (-1, 2))
        temp_img_size = (401 , 151)
        temp_img = np.zeros(temp_img_size, dtype = np.int32)
        heightdiff = 0.1
        indices = ((tempData[:,0] >= 0) & (tempData[:,0] < 151) & (tempData[:,1] >= 0) & (tempData[:,1] < 401))
        tempData = tempData[np.where(indices == True)]
        pointstmp = points[np.where(indices == True)]
        pointstmp[:,:2] = tempData[:,:2]
        df = pd.DataFrame({'row' : pointstmp[:,1].astype(np.int32), 'col' : pointstmp[:,0].astype(np.int32), 'z' : pointstmp[:,2]})
        groupmax = df.groupby(['row', 'col']).max().reset_index().values
        groupmin = df.groupby(['row', 'col']).min().reset_index().values
        groupdata = groupmin.copy()
        groupdata[:,2] = groupmax[:,2] - groupmin[:,2]
        indexz = groupdata[:,2] > heightdiff
        groupdata = groupdata[np.where(indexz == True)][:,:2].astype(np.int32)
        temp_img[tuple(groupdata.T)] = (2) # 00000010 laser obstacle
            
        lasermsg = structFUSIONMAP()
        lasermsg.resolution = 0.2
        lasermsg.cols = 151
        lasermsg.rows = 401
        lasermsg.center_col = 75
        lasermsg.center_row = 300
        lasermsg.cells = temp_img.tolist()
        if xcm_version == "LCM":
            tunnel.publish("FUSIONMAP",lasermsg.encode())
        elif xcm_version == "ZCM":
            tunnel.publish("FUSIONMAP",lasermsg)

# ==============================================================================
# -- LaneDetector --------------------------------------------------------------
# ==============================================================================

#TODO Debug with some waypoints get None and some slices get None
class LaneDetector(object):
    def __init__(self, player):
        self._parent = player
        self.location = self._parent.get_location()
        world = self._parent.get_world()
        self.current_waypoint = world.get_map().get_waypoint(self.location, project_to_road=True, lane_type=(
                carla.LaneType.Driving | carla.LaneType.Shoulder | carla.LaneType.Sidewalk))
        self.left_num = 0
        self.right_num = 0
        self.current_slice = None
        self.waypoints = None
        self.slices = None
        self.linepoints = None

    def generate_waypoints(self, distances, worldmap):
        self.location = self._parent.get_location()
        self.current_waypoint = worldmap.get_waypoint(self.location, project_to_road=True, lane_type=(
                carla.LaneType.Driving | carla.LaneType.Shoulder | carla.LaneType.Sidewalk))
        temp = list(map(self.get_one_waypoint, distances))
        for i in range(len(temp)):
            if temp[i]==None:
                temp.pop(i)
        self.waypoints = temp
        #print('waypoints number: %s' % len(self.waypoints))
        self.slices = list(map(self.get_one_slice, self.waypoints))
        #print('slices %s'%len(self.slices))
        self.current_slice = self.get_one_slice(self.current_waypoint)
        #print('current slice %s' % len(self.current_slice))

    def calculate_line_points(self):
        waypoints = []
        linepoints = []
        slice0 = self.current_slice
        temp = []
        for i in range(len(slice0)):
            temp.append(complex(slice0[i].transform.location.x, slice0[i].transform.location.y))
        waypoints.append(temp)
        for slice in self.slices:
            temp = []
            for i in range(len(slice)):
                temp.append(complex(slice[i].transform.location.x, slice[i].transform.location.y))
            waypoints.append(temp)
        #print("waypoints: %s \n"%waypoints)
        for i in range(len(waypoints)-1):
            temp = []
            temp0 = []
            temp1 = []
            for a,b in zip(waypoints[i], waypoints[i+1]):
                mid = ( a + b ) / 2.0
                temp.append(mid)
                if i==0:
                    offset = ( a - b ) / 2.0
                    temp0.append(a + offset)
                if i==len(waypoints)-2:
                    offset = ( b - a ) / 2.0
                    temp1.append(b + offset)
            #print(temp)
            #print(temp0)
            #print(temp1)
            if i==0:
                linepoints.append(temp0)
            elif i<len(waypoints)-2:
                linepoints.insert(i+1, temp)
            elif i==len(waypoints)-2:
                linepoints.append(temp1)
        #print("linepoints: %s" % len(linepoints))

        vehicle_position = complex(self.location.x, self.location.y)
        temp2 = []
        for seq in linepoints:
            temp3 = []
            for cmpl_point in seq:
                temp3.append(cmpl_point - vehicle_position)
            temp2.append(temp1)
        self.linepoints = temp2
        #self.linepoints = [cmpl_point-vehicle_position for cmpl_point in seq for seq in linepoints]

    def pack_lanes(self, pnum):
        global lanes, tunnel, xcm_version
        slice = self.current_slice
        x0 = self.location.x
        y0 = self.location.y
        right_num = self.right_num
        left_num = self.left_num
        current_lane_id = right_num + 1
        num = right_num + left_num + 1
        lanesdata = []
        # const value for line_type
        # TYPE_SOLID = 0x00;
        # TYPE_DASHED = 0x01;
        # TYPE_WHITE = 0x00;
        # TYPE_YELLOW = 0x02;
        # TYPE_SOLID_WHITE = 0x00;
        # TYPE_SOLID_YELLOW = 0x02;
        # TYPE_DASHED_WHITE = 0x01;
        # TYPE_DASHED_YELLOW = 0x03;
        # const value for lane_type const int8_t
        # TYPE_NONE = 0x00;
        # TYPE_STRAIGHT = 0x01;
        # TYPE_LEFT = 0x02;
        # TYPE_RIGHT = 0x04;
        # TYPE_UTURN = 0x08;
        # TYPE_STRAIGHT_LEFT = 0x03;
        # TYPE_STRAIGHT_RIGHT = 0x05;
        # TYPE_STRAIGHT_LEFT_RIGHT = 0x07;
        for i in range(1 + right_num + left_num):
            templane = LANE()
            templane.lane_type = LANE.TYPE_STRAIGHT_LEFT_RIGHT
            templane.width = 3.5
            templane.stop_point = LinePoint()
            templane.stop_point.x = -1.0
            templane.stop_point.y = -1.0
            # left lane line
            left_line = LaneLine()
            left_line.line_type = LaneLine.TYPE_SOLID
            left_line.distance = 0.2
            left_line.num = pnum
            points = []
            for p in self.linepoints[i+1]:
                point = LinePoint()
                point.x = p.real
                point.y = p.imag
                points.append(point)
            left_line.points = points
            templane.left_line = left_line
            # right lane line
            right_line = LaneLine()
            right_line.distance = 0.2
            right_line.line_type = LaneLine.TYPE_SOLID
            right_line.num = pnum
            points = []
            for p in self.linepoints[i]:
                point = LinePoint()
                point.x = p.real
                point.y = p.imag
                points.append(point)
            right_line.points = points
            templane.right_line = right_line
            lanesdata.append(templane)

        lanes.current_lane_id = current_lane_id
        lanes.num = num
        lanes.lanes = lanesdata
        if xcm_version=="ZCM":
            tunnel.publish("LANES", lanes)
            print("published LANES")
        elif xcm_version=="LCM":
            tunnel.publish("LANES", lanes.encode())
            print("published LANES")

    def print_waypoint(self):
        print("Current lane type: " + str(self.current_waypoint.lane_type))
        # Check current lane change allowed
        print("Current Lane change:  " + str(self.current_waypoint.lane_change))
        # Left and Right lane markings
        print("L lane marking type: " + str(self.current_waypoint.left_lane_marking.type))
        print("L lane marking change: " + str(self.current_waypoint.left_lane_marking.lane_change))
        print("R lane marking type: " + str(self.current_waypoint.right_lane_marking.type))
        print("R lane marking change: " + str(self.current_waypoint.right_lane_marking.lane_change))

    def get_one_waypoint(self, distance):
        temp = self.current_waypoint.next(distance)
        #print(type(temp))
        #print(len(temp))
        if not temp:
            return None
        return self.current_waypoint.next(distance)[0]

    def get_one_slice(self, wp):
        #print("execute get_one_slice")
        #print(wp)
        left_ptr = 0
        right_ptr = 0
        slice = []  # [R1, R2, R3, current, L1, L2, L3]
        slice.append(wp)
        wpl = wp
        wpr = wp
        if wp==None: # when central waypoint doesn't exist
            #print("exit for waypoint None , in slices")
            self.left_num = left_ptr
            self.right_num = right_ptr
            return None
        while True:
            #print(wp)
            wpl = wpl.get_left_lane()
            #print("wpl: %s"%wpl)
            if not wpl:
                break # when no lane exists, return
            if left_ptr>=1: # to avoid central line crossing error
                break
            left_ptr += 1
            slice.append(wpl)
        while True:
            wpr = wpr.get_right_lane()
            if not wpr:
                break
            if right_ptr>=1:
                break
            right_ptr += 1
            slice.insert(right_ptr-1, wpr)
        #slice.append(left_ptr)
        #slice.append(right_ptr)
        self.left_num = left_ptr
        self.right_num = right_ptr
        return  slice


# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================


def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(2.0)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)
        #world = World(client.load_world('Town01'), hud, args)
        world = World(client.get_world(), hud, args)
        controller = KeyboardControl(world, args.autopilot)
        hud.pid_controller = controller.pid_controller

        GnssSensor.start_pub_navinfo(200.0) # async publish
        GnssSensor.start_pub_caninfo(200.0)
        #CameraManager.start_pub_fusionmap()
        start_sub()

        clock = pygame.time.Clock()
        while True:

            clock.tick_busy_loop(60.0)
            if controller.parse_events(client, world, clock):
                return
            #world.pub_objectlist()
            world.tick(clock)
            world.render(display)
            pygame.display.flip()

    finally:

        if (world and world.recording_enabled):
            client.stop_recorder()

        if world is not None:
            world.destroy()

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
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
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()

