import numpy as np 
from SuperSafety.Utils.TD3 import TD3
from SuperSafety.Utils.HistoryStructs import TrainHistory
from SuperSafety.Utils.RewardFunctions import *
import torch
from numba import njit

# @njit(cache=True)
# def calculate_max_speed(delta):
#     b = 0.523
#     g = 9.81
#     l_d = 0.329
#     f_s = 0.5
#     max_v = 7

#     if abs(delta) < 0.06:
#         return max_v

#     V = f_s * np.sqrt(b*g*l_d/np.tan(abs(delta)))

#     return V

class BaseVehicle: 
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

    def transform_obs(self, obs):
        """
        Transforms the observation received from the environment into a vector which can be used with a neural network.
    
        Args:
            obs: observation from env

        Returns:
            nn_obs: observation vector for neural network
        """
        ego_idx = obs['ego_idx']
        v_current = obs['linear_vels_x'][ego_idx]
        d_current = obs['steering_deltas'][ego_idx]
        scan = np.array(obs['scans'][ego_idx]) 

        scan = np.clip(scan/self.range_finder_scale, 0, 1)
        cur_v = [v_current/self.max_v]
        cur_d = [d_current/self.max_steer]

        nn_obs = np.concatenate([cur_v, cur_d, scan])

        return nn_obs

    def transform_action(self, nn_action):
        steering_angle = nn_action[0] * self.max_steer
        # this is to ensure that it doesn't stay still
        # speed = (nn_action[1] + 1) * (self.max_v  / 2 - 0.5) + 1
        # max_speed = calculate_speed(steering_angle)
        # speed = self.max_v
        action = np.array([steering_angle, self.speed])

        return action

class TrainVehicle(BaseVehicle):
    def __init__(self, agent_name, sim_conf, load=False):
        super().__init__(agent_name, sim_conf)

        self.path = sim_conf.vehicle_path + agent_name
        state_space = 2 + self.n_beams
        self.agent = TD3(state_space, 1, 1, agent_name)
        self.agent.try_load(load, sim_conf.h_size, self.path)

        self.state = None
        self.nn_state = None
        self.nn_act = None
        self.action = None

        self.t_his = TrainHistory(agent_name, sim_conf, load)

        # self.calculate_reward = RefDistanceReward(sim_conf) 
        self.calculate_reward = RefCTHReward(sim_conf) 

    def plan(self, obs, add_mem_entry=True):
        nn_obs = self.transform_obs(obs)
        if add_mem_entry:
            self.add_memory_entry(obs, nn_obs)
            
        if obs['linear_vels_x'][0] < self.v_min_plan:
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
            reward = self.calculate_reward(self.state, s_prime)
    
            self.t_his.add_step_data(reward)

            self.agent.replay_buffer.add(self.nn_state, self.nn_act, nn_s_prime, reward, False)

    def done_entry(self, s_prime):
        """
        To be called when ep is done.
        """
        nn_s_prime = self.transform_obs(s_prime)
        reward = self.calculate_reward(self.state, s_prime)

        self.t_his.add_step_data(reward)
        self.t_his.lap_done(False)
        # self.t_his.print_update(False) #remove this line
        if self.t_his.ptr % 10 == 0:
            self.t_his.print_update(False)
        self.agent.save(self.path)
        self.state = None

        self.agent.replay_buffer.add(self.nn_state, self.nn_act, nn_s_prime, reward, True)

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


class TestVehicle(BaseVehicle):
    def __init__(self, agent_name, sim_conf):
        """
        Testing vehicle using the reference modification navigation stack

        Args:
            agent_name: name of the agent for saving and reference
            sim_conf: namespace with simulation parameters
            mod_conf: namespace with modification planner parameters
        """

        super().__init__(agent_name, sim_conf)

        self.path = sim_conf.vehicle_path + agent_name
        self.actor = torch.load(self.path + '/' + agent_name + "_actor.pth")

        print(f"Agent loaded: {agent_name}")

    def plan(self, obs):
        nn_obs = self.transform_obs(obs)

        if obs['linear_vels_x'][0] < self.v_min_plan:
            self.action = np.array([0, 7])
            return self.action

        nn_obs = torch.FloatTensor(nn_obs.reshape(1, -1))
        nn_action = self.actor(nn_obs).data.numpy().flatten()
        self.nn_act = nn_action

        self.action = self.transform_action(nn_action)

        return self.action # implemented for the safety wrapper

