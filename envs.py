import cv2
import time
import numpy as np

import gym
from gym import spaces
from gym.spaces.box import Box

import universe
from universe import vectorized
from universe import spaces as vnc_spaces
from universe.spaces.vnc_event import keycode
from universe.wrappers import BlockingReset, GymCoreAction, Unvectorize, Vectorize, Vision

from skimage.color import rgb2gray

## Factory
def create_env(type_name, env_id, client_id, remotes, **kwargs):
    type_name = type_name.lower()
    if type_name == 'atari':
        return create_atari_env(env_id)
    elif type_name == 'vnc_atari':
        return create_vncatari_env(env_id, client_id, remotes, **kwargs)
    elif type_name == 'flashgames':
        return create_flash_env(env_id, client_id, remotes, **kwargs)
    else:
        print('Unknown game type: [{}]. Try [atari|vnc_atari|flashgames]'.format(type_name))

## Atari
def create_atari_env(env_id):
    env = gym.make(env_id)
    env = Vectorize(env)
    env = AtariRescale(env)
    env = NormalizedEnv(env)
    env = Unvectorize(env)
    return env

## VNC Atari
def create_vncatari_env(env_id, client_id, remotes, **_):
    env = gym.make(env_id)
    env = Vision(env)
    env = BlockingReset(env)
    env = GymCoreAction(env)
    env = AtariRescale(env)
    env = Unvectorize(env)

    print('Connecting to remotes: {}'.format(remotes))
    env.configure(remotes=remotes, start_timeout=15 * 60, fps=env.metadata['video.frames_per_second'], client_id=client_id)
    return env

class AtariRescale(vectorized.ObservationWrapper):
    def __init__(self, env=None):
        super(AtariRescale, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [1, 80, 80])

    def _observation(self, observation_n):
        return [_process_frame(observation) for observation in observation_n]

def _process_frame(frame):
    frame = frame[34:34+160, :160]
    frame = cv2.resize(rgb2gray(frame), (80, 80))
    frame = cv2.resize(frame, (80, 80))
    frame = np.reshape(frame, [1, 80, 80])
    return frame

class NormalizedEnv(vectorized.ObservationWrapper):

    def __init__(self, env=None):
        super(NormalizedEnv, self).__init__(env)
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0

    def _observation(self, observation_n):
        for observation in observation_n:
            self.num_steps += 1
            self.state_mean = self.state_mean * self.alpha + \
                observation.mean() * (1 - self.alpha)
            self.state_std = self.state_std * self.alpha + \
                observation.std() * (1 - self.alpha)

        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))

        return [(observation - unbiased_mean) / (unbiased_std + 1e-8) for observation in observation_n]

## Flash Games
def create_flash_env(env_id, client_id, remotes, **_):
    env = gym.make(env_id)
    env = Vision(env)
    env = BlockingReset(env)

    reg = universe.runtime_spec('flashgames').server_registry
    height = reg[env_id]["height"]
    width = reg[env_id]["width"]
    env = CropScreen(env, height, width, 84, 18)
    env = FlashRescale(env)

    keys = ['left', 'right', 'up', 'down', 'x']
    if env_id == 'flashgames.NeonRace-v0':
        # Better key space for this game.
        keys = ['left', 'right', 'up', 'left up', 'right up', 'down', 'up x']
    print('Create Flash Game [{}]: keys={}'.format(env_id, keys))

    env = DiscreteToFixedKeysVNCActions(env, keys)
    env = Unvectorize(env)
    env.configure(fps=5.0, remotes=remotes, start_timeout=15 * 60, client_id=client_id,
                  vnc_driver='go', vnc_kwargs={
                    'encoding': 'tight', 'compress_level': 0,
                    'fine_quality_level': 50, 'subsample_level': 3})
    return env

class CropScreen(vectorized.ObservationWrapper):
    """Crops out a [height]x[width] area starting from (top,left) """
    def __init__(self, env, height, width, top=0, left=0):
        super(CropScreen, self).__init__(env)
        self.height = height
        self.width = width
        self.top = top
        self.left = left
        self.observation_space = Box(0, 255, shape=(height, width, 3))

    def _observation(self, observation_n):
        return [ob[self.top:self.top+self.height, self.left:self.left+self.width, :] if ob is not None else None
                for ob in observation_n]

class FlashRescale(vectorized.ObservationWrapper):
    def __init__(self, env=None):
        super(FlashRescale, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [1, 128, 200])

    def _observation(self, observation_n):
        return [_process_frame_flash(observation) for observation in observation_n]

def _process_frame_flash(frame):
    frame = cv2.resize(frame, (200, 128))
    frame = frame.mean(2).astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = np.reshape(frame, [1, 128, 200])
    return frame

class DiscreteToFixedKeysVNCActions(vectorized.ActionWrapper):
    """
    Define a fixed action space. Action 0 is all keys up. Each element of keys can be a single key or a space-separated list of keys
    """
    def __init__(self, env, keys):
        super(DiscreteToFixedKeysVNCActions, self).__init__(env)

        self._keys = keys
        self._generate_actions()
        self.action_space = spaces.Discrete(len(self._actions))

    def _generate_actions(self):
        uniq_keys = {cur_key for key in self._keys for cur_key in key.split(' ')}

        self._actions = [[vnc_spaces.KeyEvent.by_name(cur_key, down=(cur_key in key.split(' ')))
                            for cur_key in uniq_keys]
                                for key in [''] + self._keys]
        self.key_state = FixedKeyState(uniq_keys)

    def _action(self, action_n):
        # Each action might be a length-1 np.array. Cast to int to
        # avoid warnings.
        return [self._actions[int(action)] for action in action_n]

class FixedKeyState(object):
    def __init__(self, keys):
        self._keys = [keycode(key) for key in keys]
        self._down_keysyms = set()

    def apply_vnc_actions(self, vnc_actions):
        for event in vnc_actions:
            if isinstance(event, vnc_spaces.KeyEvent):
                if event.down:
                    self._down_keysyms.add(event.key)
                else:
                    self._down_keysyms.discard(event.key)

    def to_index(self):
        action_n = 0
        for key in self._down_keysyms:
            if key in self._keys:
                # If multiple keys are pressed, just use the first one
                action_n = self._keys.index(key) + 1
                break
        return action_n