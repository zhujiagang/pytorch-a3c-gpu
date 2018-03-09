from __future__ import print_function

import os
import json
import datetime
import argparse

from setproctitle import setproctitle

import torch
import torch.nn.functional as F
from torch.autograd import Variable

import gym
from gym.wrappers import Monitor

from model import ActorCriticLSTM

from envs import create_env
from eval_utils import play_game

def main():
    # Play settings
    parser = argparse.ArgumentParser(description='A3C:Play')
    parser.add_argument('--name', type=str, default='1', required=False, help="Experiment name. All outputs will be stored in checkpoints/[name]/")
    parser.add_argument('--model_name', default='best_model', help='Model to play with (defualt: best_model)')

    parser.add_argument('--seed', type=int, default=1, help='Random seed (default: 1)')

    parser.add_argument('--n_eps', default=100, help='# of epsisode (default: 100)')
    parser.add_argument('--gpu_id', default=0, help='GPU id (default: 0)')

    parser.add_argument('--no_render', action='store_true', help='Do not render to screen (default: False)')
    parser.add_argument('--random', action='store_true', help='Act randomly (default: False)')
    parser.add_argument('--duration', type=float, default=5, help='How long does the play last (default: 5 [min])')
    args = parser.parse_args()

    args.save_path = os.path.join('checkpoints', args.name)
    args.model_path = os.path.join(args.save_path, 'snapshots', '{}.pth'.format(args.model_name))
    args.gif_path = os.path.join(args.save_path, 'gifs', '{}_{}'.format(args.model_name, 
                            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))

    with open(os.path.join(args.save_path, 'config')) as f:
        vargs = json.loads(''.join(f.readlines()))
    vargs.update(vars(args))
    args.__dict__ = vargs

    print('------------ Options -------------')
    for k, v in sorted(vars(args).items()):
        print('{}: {}'.format(k, v))
    print('-------------- End ----------------')

    if not os.path.isdir(args.gif_path):
        os.makedirs(args.gif_path)

    setproctitle('{}:play'.format(args.name))

    torch.manual_seed(args.seed)
    env = create_env(args.game_type, args.env_name, 'play', 1)
    env = Monitor(env, args.gif_path, force=True)
    env._max_episode_seconds = args.duration*60
    env.seed(args.seed)

    model = ActorCriticLSTM(env.observation_space.shape[0], env.action_space.n)
    model.load_state_dict(torch.load(args.model_path))
    if args.gpu_id >= 0:
        with torch.cuda.device(args.gpu_id):
            model.cuda()

    model.eval()

    best_reward = None
    for eps in range(args.n_eps):
        model.reset()
        reward, _ = play_game(env, model, render=not args.no_render, rand=args.random, gpu_id=args.gpu_id)
        best_reward = reward if best_reward is None else max(best_reward, reward)
        print('EPS: {}/{}, Reward: {}'.format(eps+1, args.n_eps, reward))

    env.close()

    if args.n_eps > 10:
        gym.upload(args.gif_path, api_key='sk_aQXs9Po5RUyv0ZDQnkZ2A')
    os.rename(args.gif_path, args.gif_path+'_'+str(best_reward))

if __name__ == '__main__':
    main()