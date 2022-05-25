import numpy as np 
from RacingRewards.Utils.DQN import DQN
from RacingRewards.Utils.PPO import PPO
from RacingRewards.Utils.HistoryStructs import TrainHistory
from RacingRewards.Utils.RewardFunctions import *
import torch
from numba import njit

from RacingRewards.Utils.utils import init_file_struct, calculate_speed

from matplotlib import pyplot as plt



class BaseVehicleDiscrete: 
    def __init__(self, agent_name, sim_conf):
        self.name = agent_name
        self.n_beams = sim_conf.n_beams
        self.max_v = sim_conf.max_v
        self.speed = sim_conf.vehicle_speed
        self.max_steer = sim_conf.max_steer
        self.range_finder_scale = sim_conf.range_finder_scale

        self.loop_counter = 0
        self.action = None
        self.v_min_plan =  sim_conf.v_min_plan

        n_steer = 5
        self.n_steer = n_steer
        self.actions = np.zeros((n_steer, 2))
        self.actions[:, 0] = np.linspace(-0.4, 0.4, n_steer)
        self.actions[:, 1] = 2

        print(self.actions)
        print("Pause")

    def transform_obs(self, obs):
        """
        Transforms the observation received from the environment into a vector which can be used with a neural network.
    
        Args:
            obs: observation from env

        Returns:
            nn_obs: observation vector for neural network
        """

        v_current = obs['state'][3]
        d_current = obs['state'][4]
        scan = np.array(obs['scan']) 

        scan = np.clip(scan/self.range_finder_scale, 0, 1)

        cur_v = [v_current/self.max_v]
        cur_d = [d_current/self.max_steer]

        nn_obs = np.concatenate([cur_v, cur_d, scan])

        return nn_obs

    def transform_action(self, nn_action):
        action = self.actions[nn_action]

        return action


class TrainVehicleDiscrete(BaseVehicleDiscrete):
    def __init__(self, run, sim_conf):
        super().__init__(run.run_name, sim_conf)

        self.path = sim_conf.vehicle_path + run.path + run.run_name 
        init_file_struct(self.path)
        state_space = 2 + self.n_beams
        self.agent = PPO(state_space, self.n_steer, run.run_name)
        # self.agent = DQN(state_space, self.n_steer, run.run_name)
        # self.agent.create_agent(sim_conf.h_size)
        self.agent.create_agent(100)

        self.state = None
        self.nn_state = None
        self.nn_act = None
        self.action = None

        self.t_his = TrainHistory(run, sim_conf)

    def plan(self, obs, add_mem_entry=True):
        nn_obs = self.transform_obs(obs)
        if add_mem_entry:
            self.add_memory_entry(obs, nn_obs)
            
        if obs['state'][3] < self.v_min_plan:
            self.action = np.array([0, 7])
            return self.action

        self.state = obs
        nn_action = self.agent.act(nn_obs)
        self.nn_act = nn_action

        self.nn_state = nn_obs

        self.action = self.transform_action(nn_action)

        return self.action # implemented for the safety wrapper

    def add_memory_entry(self, s_prime, nn_s_prime):
        if self.state is not None:
            reward = s_prime['reward']
            
            self.t_his.add_step_data(reward)

            # self.agent.replay_buffer.add(self.nn_state, self.nn_act, nn_s_prime, reward, False)
            self.agent.put_action_data(self.nn_state, self.nn_act, nn_s_prime, reward, False)

    def done_entry(self, s_prime):
        """
        To be called when ep is done.
        """
        nn_s_prime = self.transform_obs(s_prime)
        # reward = self.calculate_reward(self.state, s_prime)
        reward = s_prime['reward']

        self.t_his.add_step_data(reward)
        self.t_his.lap_done(False)
        # self.t_his.print_update(False) #remove this line
        if self.t_his.ptr % 10 == 0:
            self.t_his.print_update(False)
        self.agent.save(self.path)
        self.state = None

        # self.agent.replay_buffer.add(self.nn_state, self.nn_act, nn_s_prime, reward, True)
        self.agent.put_action_data(self.nn_state, self.nn_act, nn_s_prime, reward, True)

    def intervention_entry(self, s_prime):
        """
        To be called when the supervisor intervenes
        """
        nn_s_prime = self.transform_obs(s_prime)
        reward = self.calculate_reward(self.state, s_prime)

        self.t_his.add_step_data(reward)

        self.agent.replay_buffer.add(self.nn_state, self.nn_act, nn_s_prime, reward, True)

    def lap_complete(self):
        """
        To be called when ep is done.
        """
        self.t_his.lap_done(False)
        self.t_his.print_update(False) #remove this line
        if self.t_his.ptr % 10 == 0:
            self.t_his.print_update(False)
            self.agent.save(self.path)


class TestVehicleDiscrete(BaseVehicleDiscrete):
    def __init__(self, run, sim_conf):
        """
        Testing vehicle using the reference modification navigation stack

        Args:
            agent_name: name of the agent for saving and reference
            sim_conf: namespace with simulation parameters
            mod_conf: namespace with modification planner parameters
        """

        super().__init__(run.run_name, sim_conf)
        self.path = sim_conf.vehicle_path + run.path + run.run_name 

        self.model = torch.load(self.path + '/' + run.run_name + "_model.pth")

        print(f"Agent loaded: {run.run_name}")

    def plan(self, obs):
        nn_obs = self.transform_obs(obs)

        if obs['state'][3] < self.v_min_plan:
            self.action = np.array([0, 7])
            return self.action

        nn_action = self.model.get_action(nn_obs)

        self.nn_act = nn_action

        self.action = self.transform_action(nn_action)

        return self.action # implemented for the safety wrapper


