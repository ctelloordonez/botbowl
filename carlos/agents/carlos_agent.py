from ctypes import c_int
import os
import gym
from botbowl import NewBotBowlEnv
from torch.autograd import Variable
import botbowl
from botbowl.ai.layers import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from botbowl.ai.new_env import EnvConf
from botbowl.core.util import get_data_path
from examples.scripted_bot_example import MyScriptedBot
 
 
# Architecture
model_name = 'epoch-28'
env_name = 'botbowl-v3'
model_filename = f"carlos/data/models/{model_name}.pth"
log_filename = f"logs/{env_name}/{env_name}.dat"
 
# Environment
 
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
 
 
class ResidualBlock(nn.Module):
    def __init__(self, in_channels=128, out_channels=128):
        super(ResidualBlock, self).__init__()
        self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        # self.bn0 = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        # self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        # self.bn2 = nn.BatchNorm2d(out_channels)
 
    def forward(self, x):
        identity = x
        x = self.conv0(x)
        # x = self.bn0(x)
        x = F.leaky_relu(x)
        x = self.conv1(x)
        # x = self.bn1(x)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        # x = self.bn2(x)
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
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
 
    def forward(self, spatial_input, attention):
        x = self.conv(spatial_input)
        sigmoid = torch.sigmoid(attention)
        sigmoid = sigmoid.view(sigmoid.size()[0], sigmoid.size()[1], 1, 1)
        x = torch.multiply(x, sigmoid)
        return x
 
debug = False
class CNNPolicy(nn.Module):
 
    def __init__(self, spatial_shape, non_spatial_inputs, actions, hidden_nodes=num_hidden_nodes, kernels=num_cnn_kernels):
        super(CNNPolicy, self).__init__()
 
        # Spatial input stream
        self.conv_init = nn.Conv2d(in_channels=spatial_shape[0], out_channels=kernels[0], kernel_size=3, stride=1, padding=1)
        # self.bn0 = BatchNorm2d(kernels[0])
        self.r0 = ResidualBlock()
        # self.bn1 = BatchNorm2d(kernels[0])
        self.r1 = ResidualBlock()
        # self.bn2 = BatchNorm2d(kernels[0])
        self.r2 = ResidualBlock()
        # self.bn3 = BatchNorm2d(kernels[0])
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
        self.linear_fc = nn.Linear(stream_size, hidden_nodes)
        # self.dropout1 = nn.Dropout(p=0.5)
 
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
        self.actor = nn.Linear(stream_size, actions)
        # self.dropout2 = nn.Dropout(p=0.8)
 
        self.reset_parameters()
 
    def reset_parameters(self): # TODO: update
        leaky_relu_gain = nn.init.calculate_gain('leaky_relu')
        self.conv_init.weight.data.mul_(leaky_relu_gain)
        self.r0.reset_parameters()
        self.r1.reset_parameters()
        self.r2.reset_parameters()
        self.r3.reset_parameters()
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
        # spatial_input = torch.reshape(spatial_input, (1, 44, 17, 28))
        # non_spatial_input = torch.reshape(non_spatial_input, (1, 116))
        x1 = self.conv_init(spatial_input)
        x1 = F.leaky_relu(x1)
        if debug: print(x1.size())
        # x1 = self.bn0(x1)
        x1 = self.r0(x1)
        if debug: print(x1.size())
        # x1 = self.bn1(x1)
        x1 = self.r1(x1)
        if debug: print(x1.size())
        # x1 = self.bn2(x1)
        x1 = self.r2(x1)
        if debug: print(x1.size())
        # x1 = self.bn3(x1)
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
 
        # Fully-connected layers
        x3 = self.linear_fc(concatenated)
        x3 = F.relu(x3)
        # x3 = self.dropout1(x3)
        if debug: print(x3.size())
 
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
        actor = self.actor(concatenated)
 
        # return value, policy
        return value, actor
 
    def act(self, spatial_inputs, non_spatial_input, action_mask):
        values, action_probs = self.get_action_probs(spatial_inputs, non_spatial_input, action_mask=action_mask)
        actions = action_probs.multinomial(1)
        return values, actions
 
    def evaluate_actions(self, spatial_inputs, non_spatial_input, actions, actions_mask):
        value, policy = self(spatial_inputs, non_spatial_input)
        actions_mask = actions_mask.view(-1, 1, actions_mask.shape[2]).squeeze().bool()
        policy[~actions_mask] = float('-inf')
        log_probs = F.log_softmax(policy, dim=1)
        probs = F.softmax(policy, dim=1)
        action_log_probs = log_probs.gather(1, actions)
        log_probs = torch.where(log_probs[None, :] == float('-inf'), torch.tensor(0.), log_probs)
        dist_entropy = -(log_probs * probs).sum(-1).mean()
        return action_log_probs, value, dist_entropy
 
    def get_action_probs(self, spatial_input, non_spatial_input, action_mask):
        values, actions = self(spatial_input, non_spatial_input)
        # Masking step: Inspired by: http://juditacs.github.io/2018/12/27/masked-attention.html
        if action_mask is not None:
            # action_mask = torch.reshape(action_mask, (1, 8117))
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
    env: NewBotBowlEnv
 
    def __init__(self, name, 
                 env_conf: EnvConf,
                 filename=model_filename, 
                 exclude_pathfinding_moves=True):
        super().__init__(name)
        self.my_team = None
        self.is_home = True
        self.action_queue = []

        self.env = NewBotBowlEnv(env_conf)
        self.env.reset()
        self.exclude_pathfinding_moves = exclude_pathfinding_moves

        spat_obs, non_spat_obs, action_mask = self.env.get_state()
        spatial_obs_space = spat_obs.shape
        non_spatial_obs_space = non_spat_obs.shape[0]
        action_space = len(action_mask)
        # MODEL
        self.policy = CNNPolicy(spatial_obs_space, non_spatial_obs_space, 
                                 hidden_nodes=num_hidden_nodes, 
                                 kernels=num_cnn_kernels, 
                                 actions=action_space)
        state_dict_file = torch.load(filename)
        self.policy.load_state_dict(state_dict_file['model_state_dict'])
        self.policy.eval()
        # self.end_setup = False
 
    def new_game(self, game, team):
        self.my_team = team
        self.opp_team = game.get_opp_team(team)
        self.is_home = self.my_team == game.state.home_team
 
    # def _flip(self, board):
    #     flipped = {}
    #     for name, layer in board.items():
    #         flipped[name] = np.flip(layer, 1)
    #     return flipped

    def act(self, game):
        if len(self.action_queue) > 0:
            return self.action_queue.pop(0)
        # if self.end_setup:
        #     self.end_setup = False
        #     return botbowl.Action(ActionType.END_SETUP)
 
        self.env.game = game
 
        # # Get observation
        # observation = self.env._observation(game)
 
        # obs = [observation]
        # spatial_obs, non_spatial_obs = self._update_obs(obs)
 
        # action_masks = self._compute_action_masks(obs)
        # action_masks = np.array(action_masks)
        # action_masks = torch.tensor(action_masks, dtype=torch.bool)
        
        # spatial_obs, non_spatial_obs, action_mask = tuple(map(torch.from_numpy, self.env.get_state()))
        spatial_obs, non_spatial_obs, action_mask = self.env.get_state()
        spatial_obs = torch.from_numpy(np.stack(spatial_obs)).float()
        non_spatial_obs = torch.from_numpy(np.stack(non_spatial_obs)).float()
        action_mask = np.array(action_mask)
        action_mask = torch.tensor(action_mask, dtype=torch.bool)


        values, actions = self.policy.act(
            Variable(spatial_obs),
            Variable(non_spatial_obs),
            Variable(action_mask))
 
        # Create action from output
        action_idx = actions[0]
        value = values[0]
        action_objects = self.env._compute_action(action_idx.numpy()[0], flip=False)
        # position = Square(x, y) if action_type in NewBotBowlEnv.positional_action_types else None
        # action = botbowl.Action(action_type, position=position, player=None)
 
        # # Let's just end the setup right after picking a formation
        # if action.action_type.name.lower().startswith('setup'):
        #     self.end_setup = True
 
        # Return action to the framework
        self.action_queue = action_objects
        return self.action_queue.pop(0)
 
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

 
    # def _compute_action_masks(self, observations):
    #     masks = []
    #     m = False
    #     for ob in observations:
    #         mask = np.zeros(self.action_space)
    #         i = 0
    #         for action_type in self.non_spatial_action_types:
    #             mask[i] = ob['available-action-types'][action_type.name]
    #             i += 1
    #         for action_type in self.spatial_action_types:
    #             if ob['available-action-types'][action_type.name] == 0:
    #                 mask[i:i+self.board_squares] = 0
    #             elif ob['available-action-types'][action_type.name] == 1:
    #                 position_mask = ob['board'][f"{action_type.name.replace('_', ' ').lower()} positions"]
    #                 position_mask_flatten = np.reshape(position_mask, (1, self.board_squares))
    #                 for j in range(self.board_squares):
    #                     mask[i + j] = position_mask_flatten[0][j]
    #             i += self.board_squares
    #         assert 1 in mask
    #         if m:
    #             print(mask)
    #         masks.append(mask)
    #     return masks
 
    # def _compute_action(self, action_idx):
    #     if action_idx < len(self.non_spatial_action_types):
    #         return self.non_spatial_action_types[action_idx], 0, 0
    #     spatial_idx = action_idx - self.num_non_spatial_action_types
    #     spatial_pos_idx = spatial_idx % self.board_squares
    #     spatial_y = int(spatial_pos_idx / self.board_dim[1])
    #     spatial_x = int(spatial_pos_idx % self.board_dim[1])
    #     spatial_action_type_idx = int(spatial_idx / self.board_squares)
    #     spatial_action_type = self.spatial_action_types[spatial_action_type_idx]
    #     return spatial_action_type, spatial_x, spatial_y
 
    # def _update_obs(self, observations):
    #     """
    #     Takes the observation returned by the environment and transforms it to an numpy array that contains all of
    #     the feature layers and non-spatial info.
    #     """
    #     spatial_obs = []
    #     non_spatial_obs = []
 
    #     for obs in observations:
    #         spatial_ob = np.stack(obs['board'].values())
 
    #         state = list(obs['state'].values())
    #         procedures = list(obs['procedures'].values())
    #         actions = list(obs['available-action-types'].values())
 
    #         non_spatial_ob = np.stack(state+procedures+actions)
 
    #         # feature_layers = np.expand_dims(feature_layers, axis=0)
    #         non_spatial_ob = np.expand_dims(non_spatial_ob, axis=0)
 
    #         spatial_obs.append(spatial_ob)
    #         non_spatial_obs.append(non_spatial_ob)
 
    #     return torch.from_numpy(np.stack(spatial_obs)).float(), torch.from_numpy(np.stack(non_spatial_obs)).float()
 
    def make_env(self, env_name):
        env = gym.make(env_name)
        return env
 
 
def load_pair():
    directory = get_data_path('tensor_dataset')
    if not os.path.exists(directory):
        os.mkdir(directory)
 
    files = os.listdir(directory)
    file = files[0]
    print(file)
 
    filename = os.path.join(directory, 'dataset.pt')
    dataset = torch.load(filename)
    pair = dataset['X'][0]
    return pair


def _make_my_copied_bot(name):
    return CopiedAgent(name=name,
                    env_conf=EnvConf(),
                    filename=model_filename,
                    exclude_pathfinding_moves=True)


# Register the bot to the framework
botbowl.register_bot('my-copied-bot', _make_my_copied_bot)


if __name__ == "__main__":
    # state_dic =torch.load(f"ffai/data/models/{model_name}.pth")
    # print(state_dic.keys())
    # Load configurations, rules, arena and teams
    config = botbowl.load_config("bot-bowl-iii")
    config.competition_mode = False
    config.pathfinding_enabled = True
    ruleset = botbowl.load_rule_set(config.ruleset)
    arena = botbowl.load_arena(config.arena)
    home = botbowl.load_team_by_filename("human", ruleset)
    away = botbowl.load_team_by_filename("human", ruleset)
    config.debug_mode = False
    
    # for e in range(29, 30):
    # model_name = f"epoch-{e}"
    # print(model_name)
    # model_filename = f"carlos/data/models/{model_name}.pth"
    # Play 100 games
    game_times = []
    wins = 0
    draws = 0
    n = 100
    is_home = True
    tds_away = 0
    tds_home = 0
    for i in range(n):

        if is_home:
            away_agent = botbowl.make_bot('random')
            home_agent = botbowl.make_bot('my-copied-bot')
        else:
            away_agent = botbowl.make_bot('my-copied-bot')
            home_agent = botbowl.make_bot("random")
        game = botbowl.Game(i, home, away, home_agent, away_agent, config, arena=arena, ruleset=ruleset)
        game.config.fast_mode = True

        # print("Starting game", (i+1))
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