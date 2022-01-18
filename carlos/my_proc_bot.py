"""
==========================
Author: Niels Justesen
Year: 2018
==========================
This module contains an example bot that takes random actions.
"""
from botbowl import Action
import botbowl
from botbowl.core.model import Agent
from botbowl.core.procedure import *
import torch
import gym
from botbowl.ai.new_env import NewBotBowlEnv, EnvConf
import uuid
from carlos.utils import get_data_path

class Pair():
    def __init__(self, obs, action):
        self.obs = obs
        # self.action_masks = action_masks
        self.action = action

    def dump(self):
        directory = get_data_path('pairs')
        # directory = get_data_path('tensor_dataset_action_mask')
        if not os.path.exists(directory):
            os.mkdir(directory)
        filename = os.path.join(directory, f"{uuid.uuid4().hex}.pt")
        my_json = self.to_json()
        # json.dump(my_json, open(filename, "w"))
        torch.save(my_json, filename)

    def to_json(self):
        return {
            'obs': self.obs,
            'actions': self.action
        }

class MyProcBot(Agent):

    def __init__(self, name):
        super().__init__(name)

        self.env = NewBotBowlEnv()
        self.env_conf = EnvConf()
        spatial_obs_space = self.env.observation_space.spaces['board'].shape
        self.board_dim = (spatial_obs_space[1], spatial_obs_space[2])
        self.board_squares = spatial_obs_space[1] * spatial_obs_space[2]
        print(len(self.env_conf.simple_action_types))
        self.non_spatial_action_types = self.env_conf.simple_action_types
        self.num_non_spatial_action_types = len(self.non_spatial_action_types)
        self.spatial_action_types = self.env_conf.positional_action_types
        self.num_spatial_action_types = len(self.spatial_action_types)
        self.num_spatial_actions = self.num_spatial_action_types * spatial_obs_space[1] * spatial_obs_space[2]
        self.action_space = self.num_non_spatial_action_types + self.num_spatial_actions


    def compute_action(self, action_idx):
        if action_idx < len(self.non_spatial_action_types):
            return self.non_spatial_action_types[action_idx], 0, 0
        spatial_idx = action_idx - self.num_non_spatial_action_types
        spatial_pos_idx = spatial_idx % self.board_squares
        spatial_y = int(spatial_pos_idx / self.board_dim[1])
        spatial_x = int(spatial_pos_idx % self.board_dim[1])
        spatial_action_type_idx = int(spatial_idx / self.board_squares)
        spatial_action_type = self.spatial_action_types[spatial_action_type_idx]
        return spatial_action_type, spatial_x, spatial_y

    def get_action_idx(self, action):
        if action.action_type in self.non_spatial_action_types:
            return self.non_spatial_action_types.index(action.action_type)

        if action.action_type in self.spatial_action_types:
            spatial_action_type_idx = self.spatial_action_types.index(action.action_type)
            spatial_idx = spatial_action_type_idx * self.board_squares
            if action.position is None and action.player is not None:
                action.position = action.player.position
            if action.position is None:
                return -1
            spatial_pos_idx = action.position.y * self.board_dim[1] + action.position.x
            return self.num_non_spatial_action_types + spatial_idx + spatial_pos_idx

        return -1

    def _update_obs(self, observations):
        """
        Takes the observation returned by the environment and transforms it to an numpy array that contains all of
        the feature layers and non-spatial info.
        """
        spatial_obs = []
        non_spatial_obs = []

        for obs in observations:
            spatial_ob = np.stack(obs['board'].values())

            state = list(obs['state'].values())
            procedures = list(obs['procedures'].values())
            actions = list(obs['available-action-types'].values())

            non_spatial_ob = np.stack(state+procedures+actions)

            # feature_layers = np.expand_dims(feature_layers, axis=0)
            non_spatial_ob = np.expand_dims(non_spatial_ob, axis=0)

            spatial_obs.append(spatial_ob)
            non_spatial_obs.append(non_spatial_ob)

        return torch.from_numpy(np.stack(spatial_obs)).float(), torch.from_numpy(np.stack(non_spatial_obs)).float()

    
    def act(self, game):
        action = self.act2(game)
        action_idx = -1
        if action is not None and game._is_action_allowed(action):
            action_idx = self.get_action_idx(action)
        if action_idx != -1:
            action_array = np.array([action_idx])
            # action_array = np.zeros(self.action_space)
            # action_array[action_idx] = 1

            action_type, x, y = self.compute_action(action_idx)
            position = Square(x, y) if action_type in self.env_conf.positional_action_types else None
            comp_action = Action(action_type, position=position, player=None)
            # print(action.action_type == comp_action.action_type and action.position == comp_action.position)
            # print(action.to_json())
            # print(comp_action.to_json())
            self.env.game = game
            observation = self.env.get_state()
            obs = [observation]
            spatial_obs, non_spatial_obs = self._update_obs(obs)
            # action_masks = self._compute_action_masks(obs)
            # action_masks = torch.tensor(action_masks, dtype=torch.bool)
            obs = {
                'spatial_obs': spatial_obs,
                'non_spatial_obs': non_spatial_obs
            }
            pair = Pair(obs=obs, action=torch.from_numpy(np.stack(action_array)).float())
            pair.dump()

        return action

    def act2(self, game):

        # Get current procedure
        proc = game.get_procedure()
        # print(type(proc))

        # Call private function
        if isinstance(proc, CoinTossFlip):
            return self.coin_toss_flip(game)
        if isinstance(proc, CoinTossKickReceive):
            return self.coin_toss_kick_receive(game)
        if isinstance(proc, Setup):
            return self.setup(game)
        if isinstance(proc, Ejection):
            return self.use_bribe(game)
        if isinstance(proc, Reroll):
            if proc.can_use_pro:
                return self.use_pro(game)
            return self.reroll(game)
        if isinstance(proc, PlaceBall):
            return self.place_ball(game)
        if isinstance(proc, HighKick):
            return self.high_kick(game)
        if isinstance(proc, Touchback):
            return self.touchback(game)
        if isinstance(proc, Turn) and proc.quick_snap:
            return self.quick_snap(game)
        if isinstance(proc, Turn) and proc.blitz:
            return self.blitz(game)
        if isinstance(proc, Turn):
            return self.turn(game)
        if isinstance(proc, MoveAction):
            return self.player_action(game)
        if isinstance(proc, MoveAction):
            return self.player_action(game)
        if isinstance(proc, BlockAction):
            return self.player_action(game)
        if isinstance(proc, PassAction):
            return self.player_action(game)
        if isinstance(proc, HandoffAction):
            return self.player_action(game)
        if isinstance(proc, BlitzAction):
            return self.player_action(game)
        if isinstance(proc, FoulAction):
            return self.player_action(game)
        if isinstance(proc, ThrowBombAction):
            return self.player_action(game)
        if isinstance(proc, Block):
            if proc.waiting_juggernaut:
                return self.use_juggernaut(game)
            if proc.waiting_wrestle_attacker or proc.waiting_wrestle_defender:
                return self.use_wrestle(game)
            return self.block(game)
        if isinstance(proc, Push):
            if proc.waiting_stand_firm:
                return self.use_stand_firm(game)
            return self.push(game)
        if isinstance(proc, FollowUp):
            return self.follow_up(game)
        if isinstance(proc, Apothecary):
            return self.apothecary(game)
        if isinstance(proc, Interception):
            return self.interception(game)

        raise Exception("Unknown procedure")

    def use_pro(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def use_juggernaut(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def use_wrestle(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def use_stand_firm(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def coin_toss_flip(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def coin_toss_kick_receive(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def setup(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def reroll(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def use_bribe(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def place_ball(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def high_kick(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def touchback(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def turn(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def quick_snap(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def blitz(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def player_action(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def block(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def push(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def follow_up(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def apothecary(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def move_action(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def block_action(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def blitz_action(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def handoff_action(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def pass_action(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def foul_action(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def throw_bomb_action(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def catch(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def interception(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def gfi(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def dodge(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")

    def pickup(self, game):
        raise NotImplementedError("This method must be overridden by non-human subclasses")


