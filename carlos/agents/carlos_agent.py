import os
from turtle import Turtle
from numpy import size
from botbowl import BotBowlEnv
from torch.autograd import Variable
import botbowl
from botbowl.ai.layers import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from botbowl.ai.env import EnvConf
from botbowl.core.util import get_data_path
from examples.scripted_bot_example import MyScriptedBot
 
 
# Config
model_name = 'epoch-21'
model_filename = f"models/botbowl-11/db11c4c2-8206-11ec-8e41-ec63d79c77d6.nn"
istraining = True
debug = False
 
 
# Architecture
num_hidden_nodes = 1024
num_cnn_kernels = [128, 64, 17]

# Search bot
class Node:
    def __init__(self, action=None, parent=None):
        self.parent = parent
        self.children = []
        self.action = action
        self.evaluations = []

    def num_visits(self):
        return len(self.evaluations)

    def visit(self, score):
        self.evaluations.append(score)

    def score(self):
        return np.average(self.evaluations)


class Pair():
    def __init__(self, obs, action):
        self.obs = obs
        # self.action_masks = action_masks
        self.action = action

    def dump(self):
        directory = get_data_path('a2c-pairs')
        if not os.path.exists(directory):
            os.mkdir(directory)
        filename = os.path.join(directory, f"{uuid.uuid4().hex}.pt")
        my_json = self.to_json()
        torch.save(my_json, filename)

    def to_json(self):
        return {
            'obs': self.obs,
            'actions': self.action
        }
 
 
class ResidualBlock(nn.Module):
    def __init__(self, in_channels=128, out_channels=128):
        super(ResidualBlock, self).__init__()
        self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
 
    def forward(self, x):
        identity = x
        x = self.conv0(x)
        x = F.leaky_relu(x)
        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        x = F.leaky_relu(x)
 
        x += identity
        return x
 
    def reset_parameters(self):
        leaky_relu_gain = nn.init.calculate_gain('leaky_relu')
        self.conv0.weight.data.mul_(leaky_relu_gain)
        self.conv1.weight.data.mul_(leaky_relu_gain)
        self.conv2.weight.data.mul_(leaky_relu_gain)
 
class AttentionConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionConvolution, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
 
    def forward(self, spatial_input, attention):
        x = self.conv(spatial_input)
        sigmoid = torch.sigmoid(attention)
        sigmoid = sigmoid.view(sigmoid.size()[0], sigmoid.size()[1], 1, 1)
        x = torch.multiply(x, sigmoid)
        return x

    # def reset_parameters(self):
    #     leaky_relu_gain = nn.init.calculate_gain('leaky_relu')
    #     self.conv.weight.data.mul_(leaky_relu_gain)
 
class CNNPolicy(nn.Module):
 
    def __init__(self, spatial_shape, non_spatial_inputs, actions, hidden_nodes=num_hidden_nodes, kernels=num_cnn_kernels):
        super(CNNPolicy, self).__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if debug: print('Device ', self.device)
 
        # Spatial input stream
        self.conv_init = nn.Conv2d(in_channels=spatial_shape[0], out_channels=kernels[0], kernel_size=3, stride=1, padding=1)
        self.r0 = ResidualBlock()
        self.r1 = ResidualBlock()
        self.r2 = ResidualBlock()
        self.r3 = ResidualBlock()
 
        self.ac0 = AttentionConvolution(spatial_shape[0] + kernels[0], kernels[1])
        self.ac1 = AttentionConvolution(kernels[1], kernels[1])
        self.ac2 = AttentionConvolution(kernels[1], kernels[2])
 
        # Non-spatial input stream
        self.linear_h0 = nn.Linear(non_spatial_inputs, hidden_nodes)
        self.linear_h1 = nn.Linear(hidden_nodes, hidden_nodes)
        self.linear_h2 = nn.Linear(hidden_nodes, hidden_nodes)
        self.linear_h3 = nn.Linear(hidden_nodes, hidden_nodes)
 
        self.linear_h4 = nn.Linear(hidden_nodes, hidden_nodes)
        self.linear_h5 = nn.Linear(hidden_nodes, hidden_nodes)
        self.linear_h6 = nn.Linear(hidden_nodes, hidden_nodes)
        self.linear_h7 = nn.Linear(hidden_nodes, hidden_nodes)
 
        # Flatten & concatenate layer
        stream_size = kernels[0] * spatial_shape[1] * spatial_shape[2]
        stream_size += hidden_nodes
        # self.linear_fc = nn.Linear(stream_size, hidden_nodes)
 
        # Linear attention layers
        self.linear_a0 = nn.Linear(hidden_nodes, kernels[1])
        self.linear_a1 = nn.Linear(hidden_nodes, kernels[1])
        self.linear_a2 = nn.Linear(hidden_nodes, kernels[2])
 
        # Non-spatial Actor
        self.linear_ha = nn.Linear(hidden_nodes, 25)  # remove 25 magic number to variable
 
        # The outputs
        self.critic = nn.Linear(hidden_nodes, 1)
        stream_size = kernels[2] * spatial_shape[1] * spatial_shape[2]
        stream_size += 25
        # self.actor = nn.Linear(stream_size, actions)
 
        # self.reset_parameters()
 
    def reset_parameters(self): # TODO: update
        leaky_relu_gain = nn.init.calculate_gain('leaky_relu')
        self.conv_init.weight.data.mul_(leaky_relu_gain)
        self.r0.reset_parameters()
        self.r1.reset_parameters()
        self.r2.reset_parameters()
        self.r3.reset_parameters()
        self.ac0.reset_parameters()
        self.ac1.reset_parameters()
        self.ac2.reset_parameters()
        self.linear_h0.weight.data.mul_(leaky_relu_gain)
        self.linear_h1.weight.data.mul_(leaky_relu_gain)
        self.linear_h2.weight.data.mul_(leaky_relu_gain)
        self.linear_h3.weight.data.mul_(leaky_relu_gain)
 
 
        self.actor.weight.data.mul_(leaky_relu_gain)
        self.critic.weight.data.mul_(leaky_relu_gain)
 
    def forward(self, spatial_input, non_spatial_input):
        """
        The forward functions defines how the data flows through the graph (layers)
        """
        # Spatial input through residual blocks
        if not istraining: spatial_input = torch.reshape(spatial_input, (1, 44, 17, 28))
        if not istraining: non_spatial_input = torch.reshape(non_spatial_input, (1, 115))
        x1 = self.conv_init(spatial_input)
        # x1 = F.leaky_relu(x1)
        if debug: print(x1.size())
        x1 = self.r0(x1)
        if debug: print(x1.size())
        x1 = self.r1(x1)
        if debug: print(x1.size())
        x1 = self.r2(x1)
        if debug: print(x1.size())
        x1 = self.r3(x1)
        if debug: print(x1.size())
 
        # Concatenate spatial input with residual blocks stream
        spatial_concatenated = torch.cat((spatial_input, x1), dim=1)  # is this right??
        if debug: print(spatial_concatenated.size())
       
        # Non-Spatial input through linear stream
        x2 = self.linear_h0(non_spatial_input)    # h0
        x2 = F.leaky_relu(x2)
        if debug: print(x2.size())
        x2 = self.linear_h1(x2)    # h1
        x2 = F.leaky_relu(x2)
        if debug: print(x2.size())
        x2 = self.linear_h2(x2)    # h2
        x2 = F.leaky_relu(x2)
        if debug: print(x2.size())
        x2 = self.linear_h3(x2)    # h3
        x2 = F.leaky_relu(x2)
        if debug: print(x2.size())
 
        # Concatenate the input streams
        flatten_x1 = x1.flatten(start_dim=1)
        flatten_x2 = x2.flatten(start_dim=1)
        concatenated = torch.cat((flatten_x1, flatten_x2), dim=1)
        if debug: print(concatenated.size())
 
        x3 = concatenated
        # Fully-connected layers
        # x3 = self.linear_fc(concatenated)
        # x3 = F.relu(x3)
        # if debug: print(x3.size())
 
        # Flatten & concatenate linear stream
        x2 = self.linear_h4(x3)    # h5
        x2 = F.leaky_relu(x2)
        x2 = self.linear_h5(x2)   # h6
        x2 = F.leaky_relu(x2)
        x2 = self.linear_h6(x2)   # h7
        x2 = F.leaky_relu(x2)
        x2 = self.linear_h7(x2)   # h8?
        x2 = F.leaky_relu(x2)
 
        # Convolutions with Channel Attention
        x4 = self.linear_a0(x2)  
        x4 = F.leaky_relu(x4)   # a0
 
        x1 = self.ac0(spatial_concatenated, x4) #AC0 output
 
        x4 = self.linear_a1(x2)
        x4 = F.leaky_relu(x4)
 
        x1 = self.ac1(x1, x4)   # AC1 output
 
        x4 = self.linear_a2(x2)
        x4 = F.leaky_relu(x4)
 
        x1 = self.ac2(x1, x4)   # AC2 output
        if debug: print(x1.size())
       
        # Output streams
        value = self.critic(x2)
 
        x5 = self.linear_ha(x2)
 
        flatten_x1 = x1.flatten(start_dim=1)
        flatten_x5 = x5.flatten(start_dim=1)
        if debug: print(flatten_x1.size(), flatten_x5.size())
        concatenated = torch.cat((flatten_x1, flatten_x5), dim=1)
        # actor = self.actor(concatenated)
        actor = concatenated
 
        return value, actor
 
    def act(self, spatial_inputs, non_spatial_input, action_mask):
        values, action_probs = self.get_action_probs(spatial_inputs, non_spatial_input, action_mask=action_mask)
        actions = action_probs.multinomial(1)
        return values, actions
 
    def evaluate_actions(self, spatial_inputs, non_spatial_input, actions, actions_mask):
        value, policy = self(spatial_inputs, non_spatial_input)
        # actions_mask = actions_mask.view(-1, 1, actions_mask.shape[2]).squeeze().bool()
        # policy[~actions_mask] = float('-inf')
        log_probs = F.log_softmax(policy, dim=1)
        probs = F.softmax(policy, dim=1)
        action_log_probs = log_probs.gather(1, actions)
        # log_probs = torch.where(log_probs[None, :] == float('-inf'), torch.tensor(0.).cuda(), log_probs)
        dist_entropy = -(log_probs * probs).sum(-1).mean()
        return action_log_probs, value, dist_entropy
 
    def get_action_probs(self, spatial_input, non_spatial_input, action_mask):
        values, actions = self(spatial_input, non_spatial_input)
        # Masking step: Inspired by: http://juditacs.github.io/2018/12/27/masked-attention.html
        if action_mask is not None:
            if not istraining: action_mask = torch.reshape(action_mask, (1, 8116))
            actions[~action_mask] = float('-inf')
        action_probs = F.softmax(actions, dim=1)
        return values, action_probs
 
    def get_action_log_probs(self, spatial_input, non_spatial_input, action_mask=None):
        values, actions = self(spatial_input, non_spatial_input)
        # Masking step: Inspired by: http://juditacs.github.io/2018/12/27/masked-attention.html
        if action_mask is not None:
            actions[~action_mask] = float('-inf')
        log_probs = F.log_softmax(actions, dim=1)
        return values, log_probs
 
 
class CopiedAgent(Agent):
    env: BotBowlEnv
 
    def __init__(self, name, 
                 env_conf: EnvConf,
                 filename=model_filename):
        super().__init__(name)
        self.my_team = None
        self.is_home = True
        self.action_queue = []
        self.env = BotBowlEnv(env_conf)
        self.env.reset()

        spat_obs, non_spat_obs, action_mask = self.env.get_state()
        spatial_obs_space = spat_obs.shape
        non_spatial_obs_space = non_spat_obs.shape[0]
        action_space = len(action_mask)
        
        # MODEL
        self.policy = CNNPolicy(spatial_obs_space, non_spatial_obs_space, 
                                 hidden_nodes=num_hidden_nodes, 
                                 kernels=num_cnn_kernels, 
                                 actions=action_space)

        # self.policy = torch.load(filename)
        state_dict_file = torch.load(filename)
        self.policy.load_state_dict(state_dict_file['model_state_dict'])
        self.policy.eval()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if debug: print('Device ', self.device)
        self.policy.to(self.device)

        self.pairs = []
 
    def new_game(self, game, team):
        self.my_team = team
        self.opp_team = game.get_opp_team(team)
        self.is_home = self.my_team == game.state.home_team

    def act(self, game):
        if len(self.action_queue) > 0:
            action = self.action_queue.pop(0)
            if debug: print(action.to_json())
            return action
 
        self.env.game = game
        spatial_obs, non_spatial_obs, action_mask = self.env.get_state()
        spatial_obs = torch.from_numpy(np.stack(spatial_obs)).float().to(self.device)
        non_spatial_obs = torch.from_numpy(np.stack(non_spatial_obs)).float().to(self.device)
        action_mask = np.array(action_mask)
        action_mask = torch.tensor(action_mask, dtype=torch.bool).to(self.device)

        values, actions = self.policy.act(
            Variable(spatial_obs),
            Variable(non_spatial_obs),
            Variable(action_mask))
 
        # Create action from output
        action_idx = actions[0]
        if debug: print(action_idx)

        if action_idx != -1:
            obs = None
            pair = Pair(obs=obs, action=actions)
            self.pairs.append(pair)
        
        value = values[0]
        action_objects = self.env._compute_action(action_idx.cpu().numpy()[0], flip=False)
 
        # Return action to the framework
        self.action_queue = action_objects
        action = self.action_queue.pop(0)
        if debug: print(action.to_json())
        if not game._is_action_allowed(action):
            # for m, mask in enumerate(action_mask):
                # print(f'{m}: {mask}')
            print(action_idx)
            print(action_mask[action_idx])
        return action
 
    def end_game(self, game):
        """
        Called when a game ends.
        """
        winner = game.get_winning_team()
        print("Casualties: ", game.num_casualties())
        if winner is None:
            print("It's a draw")
        elif winner == self.my_team:
            print("I ({}) won".format(self.name))
            print(self.my_team.state.score, "-", self.opp_team.state.score)
        else:
            print("I ({}) lost".format(self.name))
            print(self.my_team.state.score, "-", self.opp_team.state.score)

        for pair in self.pairs:
            pair.dump()


def _make_my_copied_bot(name):
    return CopiedAgent(name=name,
                    env_conf=EnvConf(size=11, pathfinding=True),
                    filename="carlos/data/no_flip_models/epoch-25.pth")

def _make_my_copied_a2c(name):
    return CopiedAgent(name=name,
                    env_conf=EnvConf(size=11, pathfinding=True),
                    filename="models/botbowl-11/b81c531a-8f3e-11ec-9467-ec63d79c77d6.pth")


if __name__ == "__main__":
    # Register the bot to the framework
    botbowl.register_bot('my-copied-a2c', _make_my_copied_a2c) 
    botbowl.register_bot('my-copied-bot', _make_my_copied_bot) 
    # state_dic =torch.load(f"ffai/data/models/{model_name}.pth")
    # print(state_dic.keys())
    # Load configurations, rules, arena and teams
    config = botbowl.load_config("bot-bowl-iii")
    config.competition_mode = False
    config.pathfinding_enabled = True
    ruleset = botbowl.load_rule_set(config.ruleset, all_rules=False)
    arena = botbowl.load_arena(config.arena)
    home = botbowl.load_team_by_filename("human", ruleset)
    away = botbowl.load_team_by_filename("human", ruleset)
    config.debug_mode = False
    
    # Play 100 games
    game_times = []
    wins = 0
    draws = 0
    n = 100
    is_home = False
    tds_away = 0
    tds_home = 0
    for i in range(n):

        if is_home:
            away_agent = botbowl.make_bot('my-copied-a2c')
            home_agent = botbowl.make_bot('my-copied-a2c')
        else:
            away_agent = botbowl.make_bot('my-copied-a2c')
            home_agent = botbowl.make_bot("my-copied-bot")
        game = botbowl.Game(i, home, away, home_agent, away_agent, config, arena=arena, ruleset=ruleset)
        game.config.fast_mode = True

        print("Starting game", (i+1))
        game.init()
        # print("Game is over")

        winner = game.get_winner()
        if winner is None:
            draws += 1
        elif winner == home_agent and is_home:
            wins += 1
        elif winner == away_agent and not is_home:
            wins += 1

        tds_home += game.get_agent_team(home_agent).state.score
        tds_away += game.get_agent_team(away_agent).state.score
    
    print(f"Home/Draws/Away: {wins}/{draws}/{n-wins-draws}")
    print(f"Home TDs per game: {tds_home/n}")
    print(f"Away TDs per game: {tds_away/n}")