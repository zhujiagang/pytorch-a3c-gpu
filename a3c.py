import copy

from setproctitle import setproctitle

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from envs import create_env

from utils import ensure_shared_grads

def train(shared_model, optimizer, rank, global_steps, args):
    setproctitle('{}:train[{}]'.format(args.name, rank))

    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    env = create_env(args.game_type, args.env_name, 'train:{}'.format(rank), args.remotes[rank])
    env._max_episode_steps = args.max_episode_length
    env.seed(args.seed + rank)

    model = copy.deepcopy(shared_model)
    gpu_id = args.gpu_ids[rank]
    with torch.cuda.device(gpu_id):
        model = model.cuda() if gpu_id >= 0 else model
    model.train()
    optimizer = optimizer or optim.Adam(shared_model.parameters(), lr=args.lr)

    done = True
    try:
        while True:
            # Sync with the shared model
            with torch.cuda.device(gpu_id):
                model.load_state_dict(shared_model.state_dict())
            if done:
                with torch.cuda.device(gpu_id):
                    state = torch.from_numpy(env.reset()).float()
                    state = state.cuda() if gpu_id >= 0 else state
                model.reset()

            values, log_probs, rewards, entropies = [], [], [], []
            for step in range(args.n_steps):
                with global_steps.get_lock():
                    global_steps.value += 1

                value, logit = model(Variable(state.unsqueeze(0)))

                prob = F.softmax(logit)
                log_prob = F.log_softmax(logit)
                entropy = -(log_prob * prob).sum(1)

                action = prob.multinomial().data
                log_prob = log_prob.gather(1, Variable(action))
                
                raw_state, reward, done, _ = env.step(action.cpu().numpy())
                reward = max(min(reward, args.max_reward), args.min_reward)

                values.append(value)
                log_probs.append(log_prob)
                rewards.append(reward)
                entropies.append(entropy)

                if done:
                    break

                state = state.copy_(torch.from_numpy(raw_state).float())

            R = state.new().resize_((1, 1)).zero_()
            if not done:
                value, _ = model(Variable(state.unsqueeze(0), volatile=True), keep_same_state=True)
                R = value.data

            values.append(Variable(R))
            policy_loss, value_loss = 0, 0
            R = Variable(R)
            gae = state.new().resize_((1, 1)).zero_()
            for i in reversed(range(len(rewards))):
                R = args.gamma * R + rewards[i]
                advantage = R - values[i]
                value_loss = value_loss + 0.5 * advantage.pow(2)

                # Generalized Advantage Estimataion
                delta_t = rewards[i] + args.gamma * values[i + 1].data - values[i].data
                gae = gae * args.gamma * args.tau + delta_t
                policy_loss = policy_loss - log_probs[i] * Variable(gae) - 0.01 * entropies[i]

            model.zero_grad()
            (policy_loss + 0.5 * value_loss).backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 40)
            ensure_shared_grads(model, shared_model, gpu=gpu_id >= 0)
            optimizer.step()

            model.detach()

            if global_steps.value >= args.max_global_steps:
                break
    except Exception as e:
        raise
    finally:
        print('Trainer [{}] finished !'.format(rank))