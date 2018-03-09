import os
import copy
import time
import math
import visdom
import numpy as np
from datetime import datetime
from collections import deque

from setproctitle import setproctitle

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from envs import create_env

def play_game(env, model, max_episode_length=math.inf, vis=None, render=False, rand=False, gpu_id=-1):
    vis = False
    if vis:
        vis, window_id, fps = vis
        frame_dur = 1.0 / fps
        last_time = time.time()

    reward_sum, episode_length = 0, 0
    state = torch.from_numpy(env.reset()).float()
    with torch.cuda.device(gpu_id):
        state = state.cuda() if gpu_id >= 0 else state

    while True:
        _, logit = model(Variable(state.unsqueeze(0), volatile=True))
        prob = F.softmax(logit)
        action = prob.multinomial(1) if rand else prob.max(1)[1]
        xx = action.data.cpu().numpy()[0]
        raw_state, reward, done, _ = env.step(action.data.cpu().numpy()[0])
        reward_sum += reward

        episode_length += 1
        if vis and time.time() > last_time + frame_dur:
            vis.image(np.transpose(env.render(mode='rgb_array'), (2,0,1)), win=window_id)
            last_time = time.time()

        if render:
            env.render()

        if done:
            return reward_sum, False
        if episode_length > max_episode_length:
            return reward_sum, True

        state = state.copy_(torch.from_numpy(raw_state).float())

def save(model, rewards, args, step):
    torch.save(model.state_dict(), os.path.join(args.model_path, 'model_iter_{}.pth'.format(step)))
    torch.save(model.state_dict(), os.path.join(args.model_path, 'model_latest.pth'))

    with open(os.path.join(args.save_path, 'rewards'), 'a+') as f:
        for record in rewards:
            f.write('{}[{}]: {}\n'.format(datetime.fromtimestamp(record[0]), record[1], record[2]))
    del rewards[:]

def test(shared_model, global_steps, args):
    SEC_PER_DAY = 24*60*60

    setproctitle('{}:test'.format(args.name))
    # vis = visdom.Visdom(env=args.name)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
  
    env = create_env(args.game_type, args.env_name, 'test', 1)
    env._max_episode_steps = None
    env.seed(args.seed)

    model = copy.deepcopy(shared_model)
    gpu_id = args.gpu_ids[-1]
    with torch.cuda.device(gpu_id):
        model = model.cuda() if gpu_id >= 0 else model
    model.eval()

    # stat_win, rewards_win = vis.line(X=np.zeros(1), Y=np.zeros(1)), vis.line(X=np.zeros(1), Y=np.zeros(1))
    max_reward, rewards = None, []
    save_condition = args.save_intervel
    start_time = time.time()

    try:
        while True:
            # Sync with the shared model
            with torch.cuda.device(gpu_id):
                model.load_state_dict(shared_model.state_dict())
            restart, eval_start_time, eval_start_step = False, time.time(), global_steps.value
            results = []
            for idx in range(args.n_evals):
                model.reset()
                reward, exceed_limit = play_game(env, model, args.max_episode_length, gpu_id=gpu_id)
                if exceed_limit:
                    restart = True
                    break
                results.append(reward)

            if restart:
                continue

            eval_end_time, eval_end_step = time.time(), global_steps.value
            rewards.append((eval_start_time, eval_start_step, results))

            local_max_reward, local_min_reward, mean_reward = np.max(results), np.min(results), np.mean(results)
            if max_reward is None or max_reward < local_max_reward:
                max_reward = local_max_reward
                
            if local_max_reward >= max_reward:
                # Save model
                torch.save(model.state_dict(), os.path.join(args.model_path, 'best_model.pth'))

            time_since_start = eval_start_time - start_time
            day = time_since_start // SEC_PER_DAY
            time_since_start %= SEC_PER_DAY

            seconds_to_finish = (args.max_global_steps - eval_end_step)/(eval_end_step-eval_start_step)*(eval_end_time-eval_start_time)
            days_to_finish = seconds_to_finish // SEC_PER_DAY
            seconds_to_finish %= SEC_PER_DAY
            print("STEP:[{}|{}], Time: {}d {}, Finish in {}d {}".format(
                eval_start_step, args.max_global_steps, '%02d' % day, time.strftime("%Hh %Mm %Ss", time.gmtime(time_since_start)),
                '%02d' % days_to_finish, time.strftime("%Hh %Mm %Ss", time.gmtime(seconds_to_finish))))
            print('\tGLOBAL: max_reward: {}; LOCAL: avg_reward: {}, std_reward: {}, min_reward: {}, max_reward: {}'.format(
                max_reward, mean_reward, np.std(results), local_min_reward, local_max_reward))

            # Plot
            # vis.updateTrace(X=np.asarray([eval_start_step]), Y=np.asarray([mean_reward]), win=stat_win, name='mean')
            # vis.updateTrace(X=np.asarray([eval_start_step]), Y=np.asarray([local_max_reward]), win=stat_win, name='max')
            # vis.updateTrace(X=np.asarray([eval_start_step]), Y=np.asarray([local_min_reward]), win=stat_win, name='min')
            #
            # for idx, r in enumerate(results):
            #     vis.updateTrace(X=np.asarray([eval_start_step]), Y=np.asarray([r]), win=rewards_win, name='reward_{}'.format(idx))

            if eval_end_step > save_condition:
                save_condition += args.save_intervel
                save(shared_model, rewards, args, eval_end_step)

            if eval_end_step >= args.max_global_steps:
                break
    except Exception as e:
        raise
    finally:
        print('Evaluator Finished !!!')
        save(shared_model, rewards, args, global_steps.value)

def show(shared_model, global_steps, args):
    setproctitle('{}:show'.format(args.name))

    try:
        env = create_env(args.game_type, args.env_name, 'show', 1)
        model = copy.deepcopy(shared_model)
        gpu_id = args.gpu_ids[-2]
        with torch.cuda.device(gpu_id):
            model = model.cuda() if gpu_id >= 0 else model
        model.eval()
        
        while True:
            # Sync with the shared model
            with torch.cuda.device(gpu_id):
                model.load_state_dict(shared_model.state_dict())
            model.reset()
            play_game(env, model, args.max_episode_length, render=False, gpu_id=gpu_id)
            
            if global_steps.value >= args.max_global_steps:
                break
    except KeyboardInterrupt:
        raise
    finally:
        print('Player Finished !!!')