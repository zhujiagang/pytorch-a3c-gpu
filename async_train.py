import time
from setproctitle import setproctitle

import torch
import torch.multiprocessing as mp

from optimizer import SharedAdam

from envs import create_env
from eval_utils import test, show

def async_train(args, make_model, train):
    setproctitle('{}:main'.format(args.name))
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    env = create_env(args.game_type, args.env_name, 'main', 1)
    shared_model = make_model(env.observation_space.shape[0], env.action_space.n)
    shared_model.share_memory()

    if args.no_shared_optimizer:
        optimizer = None
    else:
        optimizer = SharedAdam(shared_model.parameters(), lr=args.lr)
        optimizer.share_memory()

    mp.set_start_method('spawn')
    global_steps = mp.Value('L', 0)

    processes = []
    processes.append(mp.Process(target=test, args=(shared_model, global_steps, args)))
    # if not args.no_render:
    #     processes.append(mp.Process(target=show, args=(shared_model, global_steps, args)))

    for rank in range(args.n_processes):
        processes.append(mp.Process(target=train, args=(shared_model, optimizer, rank, global_steps, args)))

    for p in processes:
        p.start()
        time.sleep(0.1)

    for p in processes:
        p.join()

    print('Main process finished !!!')

def train_a3c(args):
    from a3c import train
    from model import ActorCriticLSTM

    def make_model(n_in, n_out):
        return ActorCriticLSTM(n_in, n_out)

    async_train(args, make_model, train)