from __future__ import print_function

import os
os.environ['OMP_NUM_THREADS'] = '1'
import json
import argparse
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from async_train import train_a3c

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='A3C:Train')
    parser.add_argument('--name', type=str, default='1', required=False, help='Experiment name. All outputs will be stored in checkpoints/[name]/')
    parser.add_argument('--env_name', default='Breakout-v0', help='Environment to train on (default: Breakout-v0)') ##'CartPole-v0'
    parser.add_argument('--game_type', default='atari', help='Which type the game is [atari|vnc_atari|flashgames] (default: atari)')
    parser.add_argument('--remotes', type=str, default=None, help='The address of pre-existing VNC servers and '
                        'rewarders to use [e.g. --remotes vnc://localhost:5900+15900,vnc://localhost:5901+15901] (default: None)')

    parser.add_argument('--gpu_ids', default=[0], type=int, nargs='+', help='Which GPUs to use [-1 indicate CPU] (default: -1)')

    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate (default: 0.0001)')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor for rewards (default: 0.99)')
    parser.add_argument('--tau', type=float, default=1.00, help='Parameter for GAE (default: 1.00)')

    parser.add_argument('--no_clip_reward', action='store_true', help='Do not clip reward (default: False)')
    parser.add_argument('--max_reward', type=float, default=1.0, help='Clip reward to [min_reward, max_reward] (default: 1.0)')
    parser.add_argument('--min_reward', type=float, default=-1.0, help='Clip reward to [min_reward, max_reward] (default: -1.0)')
    
    parser.add_argument('--seed', type=int, default=1, help='Random seed (default: 1)')
    
    parser.add_argument('--n_processes', type=int, default=2, help='# of training processes (default: 16)')
    parser.add_argument('--n_steps', type=int, default=20, help='# of forward steps in A3C (default: 20)')
    parser.add_argument('--n_evals', type=int, default=10, help='# of evaluations in a round (default: 10)')
    parser.add_argument('--max_episode_length', type=int, default=10000, help='Max length of an episode (default: 10000)')
    parser.add_argument('--max_global_steps', type=int, default=80000000, help='Max steps to run (default: 80000000)')
    parser.add_argument('--save_intervel', type=int, default=2000000, help='Frequency of model saving (default: 2000000)')
    
    parser.add_argument('--no_shared_optimizer', action='store_true', help='Use an optimizer without shared momentum (deafult: False)')

    # parser.add_argument('--no_render', action='store_true', help='Do not render to screen (default:False)')

    args = parser.parse_args()
    
    if args.remotes is None:
        args.remotes = [1] * args.n_processes
    else:
        args.remotes = args.remotes.split(',')
        assert len(args.remotes) == args.n_processes
    args.gpu_ids = [args.gpu_ids[idx%len(args.gpu_ids)] for idx in range(args.n_processes+2)]

    if args.no_clip_reward:
        args.max_reward, args.min_reward = float('inf'), float('-inf')

    args.save_path = os.path.join('checkpoints', args.name)
    args.model_path = os.path.join(args.save_path, 'snapshots')

    print('------------ Options -------------')
    for k, v in sorted(vars(args).items()):
        print('{}: {}'.format(k, v))
    print('-------------- End ----------------')

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.isdir(args.model_path):
        os.makedirs(args.model_path)
    with open(os.path.join(args.save_path, 'config'), 'w') as f:
        f.write(json.dumps(vars(args)))

    train_a3c(args)

if __name__ == '__main__':
    main()