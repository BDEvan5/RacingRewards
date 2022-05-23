from RacingRewards.f110_gym.f110_env import F110Env
from RacingRewards.Utils.utils import *
from RacingRewards.Planners.PurePursuit import PurePursuit
from RacingRewards.Planners.AgentPlanner import TestVehicle, TrainVehicle

import numpy as np
import time

class TestSimulation():
    def __init__(self):
        self.run_data, self.test_params = setup_run_list("BaselineRepeatRuns")
        self.conf = load_conf("config_file")

        self.env = None
        self.planner = None
        
        self.n_test_laps = None
        self.lap_times = None
        self.completed_laps = None

        self.run_simulations()

    def run_simulations(self):
        for run in self.run_data:
            self.create_simulation(run)
            self.run_testing()

    def create_simulation(self, run):
        self.env = F110Env(map=run.map_name)

        # self.planner = TestVehicle(self.test_params, self.conf)
        self.planner = PurePursuit(self.conf, run)

        self.n_test_laps = self.test_params.n_test_laps
        self.lap_times = []
        self.completed_laps = 0


    def run_testing(self):
        start_time = time.time()

        for i in range(self.n_test_laps):
            observation = self.reset_simulation()

            while not observation['colision_done'] and not observation['lap_done']:
                action = self.planner.plan(observation)
                observation = self.run_step(action)
                self.env.render('human_fast')

                if observation['lap_done']:
                    print(f"Lap {i} Complete in time: {observation['current_laptime']}")
                    self.lap_times.append(observation['current_laptime'])
                    self.completed_laps += 1

                if observation['colision_done']:
                    print(f"Lap {i} Crashed in time: {observation['current_laptime']}")
                    

        print(f"Tests are finished in: {time.time() - start_time}")


    # this is an overide
    def run_step(self, action):
        sim_steps = self.conf.sim_steps

        sim_steps, done = sim_steps, False
        while sim_steps > 0 and not done:
            obs, step_reward, done, _ = self.env.step(action[None, :])
            sim_steps -= 1
        
        observation = self.build_observation(obs, done)

        return observation

    def build_observation(self, obs, done):
        observation = {}
        observation['current_laptime'] = obs['lap_times'][0]
        observation['scan'] = obs['scans'][0] #TODO: introduce slicing here
        # inds = np.arange(0, 1080, 40)
        # observation["scan"] = obs['scans'][0][inds]
        
        pose_x = obs['poses_x'][0]
        pose_y = obs['poses_y'][0]
        theta = obs['poses_theta'][0]
        linear_velocity = obs['linear_vels_x'][0]
        steering_angle = obs['steering_deltas'][0]
        state = np.array([pose_x, pose_y, theta, linear_velocity, steering_angle])

        observation['state'] = state
        observation['lap_done'] = False
        observation['colision_done'] = False

        observation['reward'] = 0.0
        if done and obs['lap_counts'][0] == 0: 
            observation['reward'] = -1.0
            observation['colision_done'] = True
        if obs['lap_counts'][0] == 1:
            observation['reward'] = 1.0
            observation['lap_done'] = True

        return observation

    def reset_simulation(self):
        reset_pose = np.zeros(3)[None, :]
        obs, step_reward, done, _ = self.env.reset(reset_pose)

        observation = self.build_observation(obs, done)

        return observation




def main():
    sim = TestSimulation()


if __name__ == '__main__':
    main()





