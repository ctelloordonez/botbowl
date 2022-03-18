from functools import partial
from multiprocessing import Process, Pipe
import random
from typing import Tuple, Iterable

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

import botbowl
from botbowl.ai.env import BotBowlEnv, RewardWrapper, EnvConf, ScriptedActionWrapper, BotBowlWrapper
from carlos.agents.carlos_agent import CopiedAgent, CNNPolicy
from carlos.utils import load_dataset, make_trainset, get_data_path
from a2c_env import A2C_Reward, a2c_scripted_actions
from botbowl.ai.layers import *

import csv

# Environment
env_size = 11
pathfinding_enabled = True
env_name = f"botbowl-{env_size}"
env_conf = EnvConf(size=env_size, pathfinding=pathfinding_enabled)


make_agent_from_model = partial(CopiedAgent,
                                env_conf=env_conf)


def make_env():
    env = BotBowlEnv(env_conf)
    # env = ScriptedActionWrapper(env, scripted_func=a2c_scripted_actions)
    env = RewardWrapper(env, home_reward_func=A2C_Reward())
    return env


# Training configuration
CRl = 40
CBc = 40
decay_factor = 0.99
batch_size = 5
num_steps = 12000000
num_processes = 1
steps_per_update = 5
learning_rate = 0.000005
gamma = 0.99
entropy_coef = 0.01
value_loss_coef = 0.5
max_grad_norm = 0.05
log_interval = 2000
save_interval = 10
ppcg = False


reset_steps = 5000  # The environment is reset after this many steps it gets stuck

# Self-play
selfplay = False  # Use this to enable/disable self-play
selfplay_window = 1
selfplay_save_steps = int(num_steps / 10)
selfplay_swap_steps = selfplay_save_steps

# Architecture
num_hidden_nodes = 1024
num_cnn_kernels = [128, 64, 17]

# When using A2CAgent, remember to set exclude_pathfinding_moves = False if you train with pathfinding_enabled = True


# Make directories
def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


ensure_dir("logs/")
ensure_dir("models/")
ensure_dir("plots/")
exp_id = str(uuid.uuid1())
log_dir = f"logs/{env_name}/"
model_dir = f"models/{env_name}/"
plot_dir = f"plots/{env_name}/"
ensure_dir(log_dir)
ensure_dir(model_dir)
ensure_dir(plot_dir)


class Memory(object):
    def __init__(self, steps_per_update, num_processes, spatial_obs_shape, non_spatial_obs_shape, action_space):
        self.spatial_obs = torch.zeros(steps_per_update + 1, num_processes, *spatial_obs_shape)
        self.non_spatial_obs = torch.zeros(steps_per_update + 1, num_processes, *non_spatial_obs_shape)
        self.rewards = torch.zeros(steps_per_update, num_processes, 1)
        self.returns = torch.zeros(steps_per_update + 1, num_processes, 1)
        action_shape = 1
        self.actions = torch.zeros(steps_per_update, num_processes, action_shape)
        self.actions = self.actions.long()
        self.masks = torch.ones(steps_per_update + 1, num_processes, 1)
        self.action_masks = torch.zeros(steps_per_update + 1, num_processes, action_space, dtype=torch.bool)

    def cuda(self):
        self.spatial_obs = self.spatial_obs.cuda()
        self.non_spatial_obs = self.non_spatial_obs.cuda()
        self.rewards = self.rewards.cuda()
        self.returns = self.returns.cuda()
        self.actions = self.actions.cuda()
        self.masks = self.masks.cuda()
        self.action_masks = self.action_masks.cuda()

    def insert(self, step, spatial_obs, non_spatial_obs, action, reward, mask, action_masks):
        self.spatial_obs[step + 1].copy_(torch.from_numpy(spatial_obs).float())
        self.non_spatial_obs[step + 1].copy_(torch.from_numpy(np.expand_dims(non_spatial_obs, axis=1)).float())
        self.actions[step].copy_(action)
        self.rewards[step].copy_(torch.from_numpy(np.expand_dims(reward, 1)).float())
        self.masks[step].copy_(mask)
        self.action_masks[step+1].copy_(torch.from_numpy(action_masks))

    def compute_returns(self, next_value, gamma):
        self.returns[-1] = next_value
        for step in reversed(range(self.rewards.shape[0])):
            self.returns[step] = self.returns[step + 1] * gamma * self.masks[step] + self.rewards[step]


def worker(remote, parent_remote, env: BotBowlWrapper, worker_id):
    parent_remote.close()

    steps = 0
    tds = 0
    tds_opp = 0
    next_opp = botbowl.make_bot('scripted')

    while True:
        command, data = remote.recv()
        if command == 'step':
            steps += 1
            action, dif = data[0], data[1]
            spatial_obs, reward, done, info = env.step(action)
            non_spatial_obs = info['non_spatial_obs']
            action_mask = info['action_mask']

            game: Game = env.game

            # PPCG
            if dif < 1.0:
                ball_carrier = game.get_ball_carrier()
                if ball_carrier and ball_carrier.team == env.game.state.home_team:
                    extra_endzone_squares = int((1.0 - dif) * 25.0)
                    distance_to_endzone = ball_carrier.position.x - 1
                    if distance_to_endzone <= extra_endzone_squares:
                        game.state.stack.push(Touchdown(env.game, ball_carrier))
                        game.set_available_actions()
                        spatial_obs, reward, done, info = env.step(None)
                        non_spatial_obs = info['non_spatial_obs']
                        action_mask = info['action_mask']

            tds_scored = game.state.home_team.state.score - tds
            tds_opp_scored = game.state.away_team.state.score - tds_opp
            tds = game.state.home_team.state.score
            tds_opp = game.state.away_team.state.score

            if done or steps >= reset_steps:
                # If we get stuck or something - reset the environment
                if steps >= reset_steps:
                    print("Max. number of steps exceeded! Consider increasing the number.")
                done = True
                env.root_env.away_agent = next_opp
                env.reset()
                spatial_obs, non_spatial_obs, action_mask = env.get_state()
                steps = 0
                tds = 0
                tds_opp = 0
            remote.send((spatial_obs, non_spatial_obs, action_mask, reward, tds_scored, tds_opp_scored, done))

        elif command == 'reset':
            steps = 0
            tds = 0
            tds_opp = 0
            env.root_env.away_agent = next_opp
            env.reset()
            spatial_obs, non_spatial_obs, action_mask = env.get_state()
            remote.send((spatial_obs, non_spatial_obs, action_mask, 0.0, 0, 0, False))

        elif command == 'swap':
            next_opp = data
        elif command == 'close':
            break


class VecEnv:
    def __init__(self, envs):
        """
        envs: list of botbowl environments to run in subprocesses
        """
        self.closed = False
        nenvs = len(envs)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])

        self.ps = [Process(target=worker, args=(work_remote, remote, env, envs.index(env)))
                   for (work_remote, remote, env) in zip(self.work_remotes, self.remotes, envs)]

        for p in self.ps:
            p.daemon = True  # If the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def step(self, actions: Iterable[int], difficulty=1.0) -> Tuple[np.ndarray, ...]:
        """
        Takes one step in each environment, returns the results as stacked numpy arrays
        """
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', [action, difficulty]))
        results = [remote.recv() for remote in self.remotes]
        return tuple(map(np.stack, zip(*results)))

    def reset(self, difficulty=1.0):
        for remote in self.remotes:
            remote.send(('reset', difficulty))
        results = [remote.recv() for remote in self.remotes]
        return tuple(map(np.stack, zip(*results)))

    def swap(self, agent):
        for remote in self.remotes:
            remote.send(('swap', agent))

    def close(self):
        if self.closed:
            return

        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    @property
    def num_envs(self):
        return len(self.remotes)


def main():
    # CBc = 40
    torch.cuda.empty_cache()

    env = make_env()
    env.reset()
    spat_obs, non_spat_obs, action_mask = env.get_state()
    spatial_obs_space = spat_obs.shape
    non_spatial_obs_space = non_spat_obs.shape[0]
    action_space = len(action_mask)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print('Device ', device)
    # model.load_state_dict(state_dict_file['model_state_dict'])
    # MODEL
    ac_agent = CNNPolicy(spatial_obs_space,
                         non_spatial_obs_space,
                         hidden_nodes=num_hidden_nodes,
                         kernels=num_cnn_kernels,
                         actions=action_space)
    # model_dir = get_data_path('models')
    # PATH = os.path.join(model_dir, 'carlos/data/models/epoch-29.pth')

    # ac_agent = torch.load('models/botbowl-11/87f402c6-8f07-11ec-9974-ec63d79c77d6.nn')
    # state_dict_file = torch.load('carlos/data/no_flip_models/epoch-25.pth')
    state_dict_file = torch.load('models/botbowl-11/ca18d2b4-9321-11ec-8781-ec63d79c77d6.pth')
    ac_agent.load_state_dict(state_dict_file['model_state_dict'])

    ac_agent.to(device)

    loss_function = nn.NLLLoss()
    # OPTIMIZER
    rl_optimizer = optim.RAdam(ac_agent.parameters(), lr=learning_rate)
    rl_optimizer.load_state_dict(state_dict_file['rl_optimizer_state_dict'])
    bc_optimizer = optim.RAdam(ac_agent.parameters(), lr=learning_rate)
    bc_optimizer.load_state_dict(state_dict_file['bc_optimizer_state_dict'])

    dataset = load_dataset()
 
    trainset = make_trainset(dataset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=0)

    del dataset
 
    # MEMORY STORE
    memory = Memory(steps_per_update, num_processes, spatial_obs_space, (1, non_spatial_obs_space), action_space)
    memory.cuda()
    # PPCG
    difficulty = 0.0 if ppcg else 1.0
    dif_delta = 0.01

    # Reset environments
    envs = VecEnv([make_env() for _ in range(num_processes)])

    spatial_obs, non_spatial_obs, action_masks, _, _, _, _ = map(torch.from_numpy, envs.reset(difficulty))
    non_spatial_obs = torch.unsqueeze(non_spatial_obs, dim=1)

    # Add obs to memory
    memory.spatial_obs[0].copy_(spatial_obs)
    memory.non_spatial_obs[0].copy_(non_spatial_obs)
    memory.action_masks[0].copy_(action_masks)

    # Variables for storing stats
    all_updates = 1120000
    all_episodes = 9155
    all_steps = 11200000
    rl_steps = 0
    bc_steps = 0
    bc_idx = 0
    start_bc = 0
    episodes = 0
    proc_rewards = np.zeros(num_processes)
    proc_tds = np.zeros(num_processes)
    proc_tds_opp = np.zeros(num_processes)
    episode_rewards = []
    episode_tds = []
    episode_tds_opp = []
    wins = []
    value_losses = []
    policy_losses = []
    log_updates = []
    log_episode = []
    log_steps = []
    log_win_rate = []
    log_td_rate = []
    log_td_rate_opp = []
    log_mean_reward = []
    log_difficulty = []

    with open('logs/botbowl-11/ca18d2b4-9321-11ec-8781-ec63d79c77d6.dat', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            log_updates.append(int(row[0]))
            log_episode.append(int(row[1].strip()))
            log_steps.append(int(row[2].strip()))
            log_win_rate.append(float(row[3].strip()))
            log_td_rate.append(float(row[4].strip()))
            log_td_rate_opp.append(float(row[5].strip()))
            log_mean_reward.append(float(row[6].strip()))

    # self-play
    selfplay_next_save = selfplay_save_steps
    selfplay_next_swap = selfplay_swap_steps
    selfplay_models = 0

    if selfplay:
        model_name = f"{exp_id}_selfplay_0.nn"
        model_path = os.path.join(model_dir, model_name)
        torch.save(ac_agent, model_path)
        envs.swap(make_agent_from_model(name=model_name, filename=model_path))
        selfplay_models += 1

    while all_steps < num_steps:
        torch.cuda.empty_cache()
        # if all_steps % log_interval: print(f'steps: {all_steps}')
        # if rl_steps < CRl:
        for crl in range(CRl):
            ac_agent.eval()
            for step in range(steps_per_update):
                values, actions = ac_agent.act(
                    Variable(memory.spatial_obs[step]),
                    Variable(memory.non_spatial_obs[step]),
                    Variable(memory.action_masks[step]))

                action_objects = (action[0] for action in actions.cpu().numpy())

                spatial_obs, non_spatial_obs, action_masks, shaped_reward, tds_scored, tds_opp_scored, done = envs.step(action_objects, difficulty=difficulty)

                proc_rewards += shaped_reward
                proc_tds += tds_scored
                proc_tds_opp += tds_opp_scored
                episodes += done.sum()

                # If done then clean the history of observations.
                for i in range(num_processes):
                    if done[i]:
                        if proc_tds[i] > proc_tds_opp[i]:  # Win
                            wins.append(1)
                            difficulty += dif_delta
                        elif proc_tds[i] < proc_tds_opp[i]:  # Loss
                            wins.append(0)
                            difficulty -= dif_delta
                        else:  # Draw
                            wins.append(0.5)
                            difficulty -= dif_delta
                        if ppcg:
                            difficulty = min(1.0, max(0, difficulty))
                        else:
                            difficulty = 1
                        episode_rewards.append(proc_rewards[i])
                        episode_tds.append(proc_tds[i])
                        episode_tds_opp.append(proc_tds_opp[i])
                        proc_rewards[i] = 0
                        proc_tds[i] = 0
                        proc_tds_opp[i] = 0

                # insert the step taken into memory
                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])

                memory.insert(step, spatial_obs, non_spatial_obs, actions.data, shaped_reward, masks, action_masks)

            # -- TRAINING -- #
            ac_agent.train()
            # bootstrap next value
            next_value = ac_agent(Variable(memory.spatial_obs[-1], requires_grad=False), Variable(memory.non_spatial_obs[-1], requires_grad=False))[0].data

            # Compute returns
            memory.compute_returns(next_value, gamma)

            spatial = Variable(memory.spatial_obs[:-1])
            spatial = spatial.view(-1, *spatial_obs_space)
            spatial = spatial.to(device)
            non_spatial = Variable(memory.non_spatial_obs[:-1])
            non_spatial = non_spatial.view(-1, non_spatial.shape[-1])
            non_spatial = non_spatial.to(device)

            actions = Variable(torch.LongTensor(memory.actions.cpu().view(-1, 1)))
            actions = actions.to(device)
            actions_mask = Variable(memory.action_masks[:-1])
            actions_mask = actions_mask.view(-1, action_space)
            actions_mask = actions_mask.to(device)


            # Evaluate the actions taken
            action_log_probs, values, dist_entropy = ac_agent.evaluate_actions(spatial, 
                                                                                non_spatial, 
                                                                                actions, 
                                                                                actions_mask)

            values = values.view(steps_per_update, num_processes, 1)
            action_log_probs = action_log_probs.view(steps_per_update, num_processes, 1)

            # Compute loss
            advantages = Variable(memory.returns[:-1]) - values
            value_loss = advantages.pow(2).mean()
            #value_losses.append(value_loss)

            action_loss = -(Variable(advantages.data) * action_log_probs).mean()
            #policy_losses.append(action_loss)

            rl_optimizer.zero_grad()

            total_loss = (value_loss * value_loss_coef + action_loss - dist_entropy * entropy_coef)
            total_loss.backward()

            # nn.utils.clip_grad_norm_(ac_agent.parameters(), max_grad_norm)

            rl_optimizer.step()
            """
            returns = Variable(memory.returns[:-1])
            # print(returns.shape)
            returns = returns.view(-1, 1)
            # print(spatial.shape, non_spatial.shape, actions.shape, actions_mask.shape, returns.shape)
            evaluationloader = DataLoader(TensorDataset(spatial, non_spatial, actions, actions_mask, returns), 
                                            batch_size= batch_size, shuffle=True, num_workers=0)

            running_loss = 0
            ac_agent.train()
            for j, eval_data in enumerate(evaluationloader, 0):
                eval_spatial_obs, eval_non_spatial_obs, eval_actions, eval_actions_mask, eval_returns = eval_data
                eval_spatial_obs = eval_spatial_obs.to(device)
                eval_non_spatial_obs = eval_non_spatial_obs.to(device)
                eval_actions = eval_actions.to(device)
                eval_actions_mask = eval_actions_mask.to(device)

                # Evaluate the actions taken
                action_log_probs, values, dist_entropy = ac_agent.evaluate_actions(eval_spatial_obs, 
                                                                                    eval_non_spatial_obs, 
                                                                                    eval_actions, 
                                                                                    eval_actions_mask)
                # print(values.shape)
                # print(action_log_probs.shape)
                # print(eval_returns.shape)
                # values = values.view(steps_per_update, num_processes, 1)
                # action_log_probs = action_log_probs.view(steps_per_update, num_processes, 1)

                advantages = eval_returns - values
                value_loss = advantages.pow(2).mean()
                #value_losses.append(value_loss)

                # Compute loss
                action_loss = -(Variable(advantages.data) * action_log_probs).mean()
                #policy_losses.append(action_loss)

                optimizer.zero_grad()

                total_loss = (value_loss * value_loss_coef + action_loss - dist_entropy * entropy_coef)
                running_loss+=total_loss.item()
                total_loss.backward()

                # nn.utils.clip_grad_norm_(ac_agent.parameters(), max_grad_norm)

                optimizer.step()

            train_loss=running_loss/len(evaluationloader)
            print('Total Loss: %.3f'%(train_loss))   
            """
            memory.non_spatial_obs[0].copy_(memory.non_spatial_obs[-1])
            memory.spatial_obs[0].copy_(memory.spatial_obs[-1])
            memory.action_masks[0].copy_(memory.action_masks[-1])

            # Updates
            all_updates += 1
            # Episodes
            all_episodes += episodes
            episodes = 0
            # Steps
            all_steps += num_processes * steps_per_update

            # Self-play save
            if selfplay and all_steps >= selfplay_next_save:
                selfplay_next_save = max(all_steps+1, selfplay_next_save+selfplay_save_steps)
                model_name = f"{exp_id}_selfplay_{selfplay_models}.nn"
                model_path = os.path.join(model_dir, model_name)
                print(f"Saving {model_path}")
                torch.save(ac_agent, model_path)
                selfplay_models += 1

            # Self-play swap

            if selfplay and all_steps >= selfplay_next_swap:
                selfplay_next_swap = max(all_steps + 1, selfplay_next_swap+selfplay_swap_steps)
                lower = max(0, selfplay_models-1-(selfplay_window-1))
                i = random.randint(lower, selfplay_models-1)
                model_name = f"{exp_id}_selfplay_{i}.nn"
                model_path = os.path.join(model_dir, model_name)
                print(f"Swapping opponent to {model_path}")
                envs.swap(make_agent_from_model(name=model_name, filename=model_path))
        
            rl_steps += 1
        
        trainloader_iter = iter(trainloader)
        # elif bc_steps < CBc:
        for bc in range(int(CBc)):
            ac_agent.train()
            data = next(trainloader_iter)

            # get the inputs; data is a list of [inputs, labels]
            spatial_obs, non_spatial_obs, actions = data
            spatial_obs = spatial_obs.to(device)
            non_spatial_obs = non_spatial_obs.to(device)
            actions = actions.to(device)
    
            # zero the parameter gradients
            bc_optimizer.zero_grad()
    
            # forward + backward + optimize
            values, action_log_probs, = ac_agent.get_action_log_probs(spatial_obs, non_spatial_obs)
            loss = loss_function(action_log_probs, actions)
            loss.backward()

            bc_optimizer.step()
            
            running_loss = loss.item()
            _, predicted = action_log_probs.max(1)
            total = actions.size(0)
            correct = predicted.eq(actions).sum().item()

            print('Train Loss: %.3f | Accuracy: %.3f'%(running_loss,100*correct/total))

            all_steps += batch_size
            """
            for i, data in enumerate(trainloader, start_bc):
                if i >= steps_per_update + start_bc:
                    break
                # get the inputs; data is a list of [inputs, labels]
                spatial_obs, non_spatial_obs, actions = data
                # print(spatial_obs.shape)
                spatial_obs = spatial_obs.to(device)
                non_spatial_obs = non_spatial_obs.to(device)
                actions = actions.to(device)
        
                # zero the parameter gradients
                optimizer.zero_grad()
        
                # forward + backward + optimize
                values, action_log_probs, = ac_agent.get_action_log_probs(spatial_obs, non_spatial_obs)
                # action_log_probs = torch.where(action_log_probs == float('-inf'), torch.tensor(0., device=device), action_log_probs)
                # print(spatial_obs.size(), non_spatial_obs.size(), actions.size(), outputs[1].size())
                # print(actions)
                loss = loss_function(action_log_probs, actions)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = action_log_probs.max(1)
                total += actions.size(0)
                correct += predicted.eq(actions).sum().item()

                epoch_loss.append(loss.item())
                epoch_accu.append(100* predicted.eq(actions).sum().item()/actions.size(0))

                all_steps += batch_size

            """
            bc_idx += 1 
            if bc_idx >= len(trainloader): bc_idx = 0
            bc_steps += 1
        del trainloader_iter
        # CBc *= decay_factor

        # else:
        #     rl_steps = 0
        #     bc_steps = 0

        # Logging
        if all_updates % log_interval == 0 and len(episode_rewards) >= num_processes:
            td_rate = np.mean(episode_tds)
            td_rate_opp = np.mean(episode_tds_opp)
            episode_tds.clear()
            episode_tds_opp.clear()
            mean_reward = np.mean(episode_rewards)
            episode_rewards.clear()
            win_rate = np.mean(wins)
            wins.clear()
            
            log_updates.append(all_updates)
            log_episode.append(all_episodes)
            log_steps.append(all_steps)
            log_win_rate.append(win_rate)
            log_td_rate.append(td_rate)
            log_td_rate_opp.append(td_rate_opp)
            log_mean_reward.append(mean_reward)
            log_difficulty.append(difficulty)

            log = "Upd: {}, Ep: {}, Win: {:.2f}, TD: {:.2f}, TD opp: {:.2f}, Mean reward: {:.3f}, Difficulty: {:.2f}" \
                .format(all_updates, all_episodes, win_rate, td_rate, td_rate_opp, mean_reward, difficulty)
            
            log_to_file = "{}, {}, {}, {}, {}, {}, {}\n" \
                .format(all_updates, all_episodes, all_steps, win_rate, td_rate, td_rate_opp, mean_reward, difficulty)

            # Save to files
            log_path = os.path.join(log_dir, f"{exp_id}.dat")
            print(f"Save log to {log_path}")
            with open(log_path, "a") as myfile:
                myfile.write(log_to_file)
            
            print(log)

            episodes = 0
            value_losses.clear()
            policy_losses.clear()

            # Save model
            model_name = f"{exp_id}.pth"
            model_path = os.path.join(model_dir, model_name)
            torch.save({
                        'model_state_dict': ac_agent.cpu().state_dict(),
                        'rl_optimizer_state_dict': rl_optimizer.state_dict(),
                        'bc_optimizer_state_dict': bc_optimizer.state_dict()
                        }, model_path)
            ac_agent.to(device=device)
            
            # plot
            n = 3
            if ppcg:
                n += 1
            fig, axs = plt.subplots(1, n, figsize=(4*n, 5))
            axs[0].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
            axs[0].plot(log_steps, log_mean_reward)
            axs[0].set_title('Reward')
            #axs[0].set_ylim(bottom=0.0)
            axs[0].set_xlim(left=0)
            axs[1].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
            axs[1].plot(log_steps, log_td_rate, label="Learner")
            # if selfplay:
            # axs[1].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            axs[1].plot(log_steps, log_td_rate_opp, color="red", label="Opponent")
            axs[1].set_title('TD/Episode')
            axs[1].set_ylim(bottom=0.0)
            axs[1].set_xlim(left=0)
            axs[1].legend()
            axs[2].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
            axs[2].plot(log_steps, log_win_rate)
            axs[2].set_title('Win rate')            
            axs[2].set_yticks(np.arange(0, 1.001, step=0.1))
            axs[2].set_xlim(left=0)
            if ppcg:
                axs[3].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
                axs[3].plot(log_steps, log_difficulty)
                axs[3].set_title('Difficulty')
                axs[3].set_yticks(np.arange(0, 1.001, step=0.1))
                axs[3].set_xlim(left=0)
            fig.tight_layout()
            plot_name = f"{exp_id}_{'_selfplay' if selfplay else ''}.png"
            plot_path = os.path.join(plot_dir, plot_name)
            fig.savefig(plot_path)
            plt.close('all')

    model_name = f"{exp_id}.nn"
    model_path = os.path.join(model_dir, model_name)
    torch.save(ac_agent, model_path)
    envs.close()


if __name__ == "__main__":
    main()
