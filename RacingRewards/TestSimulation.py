from RacingRewards.f110_gym.f110_env import F110Env
from RacingRewards.Utils.utils import *
from RacingRewards.Planners.PurePursuit import PurePursuit
from RacingRewards.Planners.AgentPlanner import TestVehicle, TrainVehicle

import numpy as np
import time
from RacingRewards.Reward import RaceTrack, DistanceReward


class TestSimulation():
    def __init__(self, run_file: str):
        self.run_data, self.test_params = setup_run_list(run_file)
        self.conf = load_conf("config_file")

        self.env = None
        self.planner = None
        
        self.n_test_laps = None
        self.lap_times = None
        self.completed_laps = None
        self.prev_obs = None

        self.race_track = None

        # flags 
        self.show_test = False
        self.show_train = False
        self.verbose = False #TODO: this should come from the config file

    def run_testing_evaluation(self):
        for run in self.run_data:
            self.env = F110Env(map=run.map_name)

            self.planner = TestVehicle(run, self.conf)
            # self.planner = PurePursuit(self.conf, run)

            self.n_test_laps = self.test_params.n_test_laps
            self.lap_times = []
            self.completed_laps = 0

            eval_dict = self.run_testing()
            run_dict = vars(run)
            run_dict.update(eval_dict)

            save_conf_dict(run_dict)


    def run_testing(self):
        assert self.env != None, "No environment created"
        start_time = time.time()

        for i in range(self.n_test_laps):
            observation = self.reset_simulation()

            while not observation['colision_done'] and not observation['lap_done']:
                action = self.planner.plan(observation)
                observation = self.run_step(action)
                if self.show_test: self.env.render('human_fast')

                if observation['lap_done']:
                    if self.verbose: print(f"Lap {i} Complete in time: {observation['current_laptime']}")
                    self.lap_times.append(observation['current_laptime'])
                    self.completed_laps += 1

                if observation['colision_done']:
                    if self.verbose: print(f"Lap {i} Crashed in time: {observation['current_laptime']}")
                    
        print(f"Tests are finished in: {time.time() - start_time}")

        success_rate = (self.completed_laps / (self.n_test_laps) * 100)
        if len(self.lap_times) > 0:
            avg_times, std_dev = np.mean(self.lap_times), np.std(self.lap_times)
        else:
            avg_times, std_dev = 0, 0

        print(f"Crashes: {self.n_test_laps - self.completed_laps} VS Completes {self.completed_laps} --> {success_rate:.2f} %")
        print(f"Lap times Avg: {avg_times} --> Std: {std_dev}")

        eval_dict = {}
        eval_dict['success_rate'] = float(success_rate)
        eval_dict['avg_times'] = float(avg_times)
        eval_dict['std_dev'] = float(std_dev)

        return eval_dict

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
        """Build observation

        Returns 
            state:
                [0]: x
                [1]: y
                [2]: yaw
                [3]: v
                [4]: steering
            scan:
                Lidar scan beams 
            
        """
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

        observation['reward'] = self.reward(observation, self.prev_obs)

        return observation

    def reset_simulation(self):
        reset_pose = np.zeros(3)[None, :]
        obs, step_reward, done, _ = self.env.reset(reset_pose)

        observation = self.build_observation(obs, done)
        self.prev_obs = observation

        return observation

class TrainSimulation(TestSimulation):
    def __init__(self, run_file):
        super().__init__(run_file)

        self.n_train_steps = self.test_params.n_train_steps
        self.reward = None
        self.previous_observation = None

    def run_training_evaluation(self):
        for run in self.run_data:
            self.env = F110Env(map=run.map_name)

            #train
            self.race_track = RaceTrack(run.map_name)
            self.race_track.load_center_pts()
            self.reward = DistanceReward(self.race_track)

            self.planner = TrainVehicle(run, self.conf)
            self.completed_laps = 0

            self.run_training()

            #Test
            self.planner = TestVehicle(run, self.conf)
            self.n_test_laps = self.test_params.n_test_laps
            self.lap_times = []
            self.completed_laps = 0

            eval_dict = self.run_testing()
            run_dict = vars(run)
            run_dict.update(eval_dict)

            save_conf_dict(run_dict)


    def run_training(self):
        assert self.env != None, "No environment created"
        start_time = time.time()
        print(f"Starting Baseline Training: {self.planner.name}")

        lap_counter, crash_counter = 0, 0
        observation = self.reset_simulation()

        for i in range(self.n_train_steps):
            self.prev_obs = observation
            action = self.planner.plan(observation)
            observation = self.run_step(action)

            self.planner.agent.train(2)

            if self.show_train: self.env.render('human_fast')

            if observation['lap_done'] or observation['colision_done'] or self.race_track.check_done(observation):
                self.planner.done_entry(observation)

                if observation['lap_done']:
                    if self.verbose: print(f"{i}::Lap Complete {lap_counter} -> FinalR: {observation['reward']} -> LapTime {observation['current_laptime']:.2f} -> TotalReward: {self.planner.t_his.rewards[self.planner.t_his.ptr-1]:.2f}")

                    lap_counter += 1
                    self.completed_laps += 1

                elif observation['colision_done'] or self.race_track.check_done(observation):

                    if self.verbose: print(f"{i}::Lap Crashed -> FinalR: {observation['reward']} -> LapTime {observation['current_laptime']:.2f} -> TotalReward: {self.planner.t_his.rewards[self.planner.t_his.ptr-1]:.2f}")
                    crash_counter += 1

                observation = self.reset_simulation()
                    
            
        self.planner.t_his.print_update(True)
        self.planner.t_his.save_csv_data()
        self.planner.agent.save(self.planner.path)

        train_time = time.time() - start_time
        print(f"Finished Training: {self.planner.name} in {train_time} seconds")
        print(f"Crashes: {crash_counter}")


        print(f"Training finished in: {time.time() - start_time}")



def main():
    # sim = TestSimulation("BaselineRepeatRuns")
    # sim.run_testing_evaluation()

    sim = TrainSimulation("BaselineRepeatRuns")
    sim.run_training_evaluation()

if __name__ == '__main__':
    main()



