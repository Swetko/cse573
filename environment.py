"""A wrapper for engaging with the THOR environment."""

import copy
import json
import math
import os
import random
import numpy as np

from ai2thor.controller import Controller

class Environment:
    """ Abstraction of the ai2thor enviroment. """

    def __init__(self,
                 grid_size=0.25,
                 fov=90.0,
                 local_executable_path=None,
                 randomize_objects=False,
                 seed=1):

        random.seed(seed)

        self.controller = Controller()
        if local_executable_path:
            self.controller.local_executable_path = local_executable_path
        self.grid_size = grid_size
        self._reachable_points = {}
        self.fov = fov
        self.offline_data_dir = './datasets/floorplans'
        self.y = None
        self.randomize_objects = randomize_objects
        self.stored_scene_name = None

    @property
    def scene_name(self):
        return self.controller.last_event.metadata['sceneName']

    @property
    def current_frame(self):
        return self.controller.last_event.frame

    @property
    def last_event(self):
        return self.controller.last_event

    @property
    def last_action_success(self):
        return self.controller.last_event.metadata['lastActionSuccess']


    def object_is_visible(self, objId):
        objects = self.last_event.metadata['objects']
        visible_objects = [o['objectId'] for o in objects if o['visible']]
        return objId in visible_objects


    def start(self, scene_name, gpu_id):
        """ Begin the scene. """
        # self.controller.start(x_display=str(gpu_id))
        self.controller.start()
        self.controller.step({'action': 'ChangeQuality', 'quality': 'Very Low'})
        self.controller.step(
                {
                    "action": "ChangeResolution",
                    "x": 224,
                    "y": 224,
                }
            )

        self.controller.reset(scene_name)
        self.controller.step(dict(action='Initialize', gridSize=self.grid_size, fieldOfView=self.fov))
        self.y = self.controller.last_event.metadata['agent']['position']['y']
        self.controller.step(dict(action='ToggleHideAndSeekObjects'))
        self.randomize_agent_location()
        self.seed = 0
        self.controller.step(dict(action='InitialRandomSpawn', forceVisible=True, maxNumRepeats=10, randomSeed=self.seed))


    def reset(self, scene_name, change_seed=True):
        """ Reset the scene. """
        self.controller.reset(scene_name)
        self.controller.step(dict(action='Initialize', gridSize=self.grid_size, fieldOfView=self.fov))
        if change_seed:
            self.randomize_agent_location()
        else:
            self.teleport_agent_to(**self.start_state)

        self.y = self.controller.last_event.metadata['agent']['position']['y']
        self.controller.step(dict(action='ToggleHideAndSeekObjects'))
        if change_seed and self.randomize_objects:
            self.seed = random.randint(0, 1000000)
            self.controller.step(dict(action='InitialRandomSpawn', forceVisible=True, maxNumRepeats=10, randomSeed=self.seed))
        else:
            self.controller.step(dict(action='InitialRandomSpawn', forceVisible=True, maxNumRepeats=10, randomSeed=self.seed))

    def all_objects(self): 
        objects = self.controller.last_event.metadata['objects']
        return [o['objectId'] for o in objects]
    
    def fail(self):
        self.controller.last_event.metadata['lastActionSuccess'] = False
        return self.controller.last_event

    def step(self, action_dict):
        curr_state = ThorAgentState.get_state_from_evenet(event=self.controller.last_event, forced_y=self.y)
        next_state = get_next_state(curr_state, action_dict['action'], copy_state=True)
        if action_dict['action'] in ['LookUp', 'LookDown', 'RotateLeft', 'RotateRight', 'MoveAhead']:
            if next_state is None:
                self.last_event.metadata['lastActionSuccess'] = False
            else:
                event = self.controller.step(dict(action='Teleport', x=next_state.x, y=next_state.y, z=next_state.z))
                s1 = event.metadata['lastActionSuccess']
                event = self.controller.step(dict(action='Rotate', rotation=next_state.rotation))
                s2 = event.metadata['lastActionSuccess']
                event = self.controller.step(dict(action="Look", horizon=next_state.horizon))
                s3 = event.metadata['lastActionSuccess']

                if not (s1 and s2 and s3):
                    # Go back to previous state.
                    self.teleport_agent_to(curr_state.x, curr_state.y, curr_state.z, curr_state.rotation, curr_state.horizon)
                    self.last_event.metadata['lastActionSuccess'] = False
        elif action_dict['action'] != 'Done':
            return self.controller.step(action_dict)

    def teleport_agent_to(self, x, y, z, rotation, horizon):
        """ Teleport the agent to (x,y,z) with given rotation and horizon. """
        self.controller.step(dict(action='Teleport', x=x, y=y, z=z))
        self.controller.step(dict(action='Rotate', rotation=rotation))
        self.controller.step(dict(action="Look", horizon=horizon))

    def random_reachable_state(self):
        """ Get a random reachable state. """
        xyz = random.choice(self.reachable_points)
        rotation = random.choice([0, 90, 180, 270])
        horizon = random.choice([0, 30, 330])
        state = copy.copy(xyz)
        state['rotation'] = rotation
        state['horizon'] = horizon
        return state

    def randomize_agent_location(self):
        state = self.random_reachable_state()
        self.teleport_agent_to(**state)
        self.start_state = copy.deepcopy(state)
        return

    @property
    def reachable_points(self):
        """ Use the JSON file to get the reachable points. """
        if self.scene_name in self._reachable_points:
            return self._reachable_points[self.scene_name]

        points_path = os.path.join(self.offline_data_dir, self.scene_name, "grid.json")
        if not os.path.exists(points_path):
            raise IOError("Path {0} does not exist".format(points_path))
        self._reachable_points[self.scene_name] = json.load(open(points_path))
        return self._reachable_points[self.scene_name]


class ThorAgentState:
    """ Representation of a simple state of a Thor Agent which includes
        the position, horizon and rotation. """
    def __init__(self, x, y, z, rotation, horizon):
        self.x = round(x, 2)
        self.y = y
        self.z = round(z, 2)
        self.rotation = round(rotation)
        self.horizon = round(horizon)


    @classmethod
    def get_state_from_evenet(cls, event, forced_y=None):
        """ Extracts a state from an event. """
        state = cls(
            x=event.metadata['agent']['position']['x'],
            y=event.metadata['agent']['position']['y'],
            z=event.metadata['agent']['position']['z'],
            rotation=event.metadata['agent']['rotation']['y'],
            horizon=event.metadata['agent']['cameraHorizon']
        )
        if forced_y != None:
            state.y = forced_y
        return state

    def __eq__(self, other):
        """ If we check for exact equality then we get issues.
            For now we consider this 'close enough'. """
        if isinstance(other, ThorAgentState):
            return (
                self.x == other.x and
                # self.y == other.y and
                self.z == other.z and
                self.rotation == other.rotation and
                self.horizon == other.horizon
            )
        return NotImplemented

    def __str__(self):
        return '{:0.2f}|{:0.2f}|{:d}|{:d}'.format(
            self.x,
            self.z,
            round(self.rotation),
            round(self.horizon)
        )

    def position(self):
        """ Returns just the position. """
        return dict(x=self.x, y=self.y, z=self.z)


def get_next_state(state, action, copy_state=False):
    """ Guess the next state when action is taken. Note that
        this will not predict the correct y value. """
    grid_size = 0.25
    if copy_state:
        next_state = copy.deepcopy(state)
    else:
        next_state = state
    if action == 'MoveAhead':
        if next_state.rotation == 0:
            next_state.z += grid_size
        elif next_state.rotation == 90:
            next_state.x += grid_size
        elif next_state.rotation == 180:
            next_state.z -= grid_size
        elif next_state.rotation == 270:
            next_state.x -= grid_size
        elif next_state.rotation == 45:
            next_state.z += grid_size
            next_state.x += grid_size
        elif next_state.rotation == 135:
            next_state.z -= grid_size
            next_state.x += grid_size
        elif next_state.rotation == 225:
            next_state.z -= grid_size
            next_state.x -= grid_size
        elif next_state.rotation == 315:
            next_state.z += grid_size
            next_state.x -= grid_size
        else:
            raise Exception('Unknown Rotation')
    elif action == 'RotateRight':
        next_state.rotation = (next_state.rotation + 45) % 360
    elif action == 'RotateLeft':
        next_state.rotation = (next_state.rotation - 45) % 360
    elif action == 'LookUp':
        if abs(next_state.horizon) <= 1:
            return None
        next_state.horizon = next_state.horizon - 30
    elif action == 'LookDown':
        if abs(next_state.horizon - 60) <= 1 or abs(next_state.horizon - 30) <= 1:
            return None
        next_state.horizon = next_state.horizon + 30
    return next_state
