import numpy as np 

from RacingRewards.Utils.utils import *
from RacingRewards.Planners.AgentPlanner import TestVehicle
import torch
from RacingRewards.MapData import MapData

map_name = "columbia_small"

class AgentWrapped:
    def __init__(self, agent_name):
        self.agent_name = agent_name
        self.conf = load_conf("config_file")
        run = {'map_name': 'columbia_small'}

        # I need to load the value network in, not the agent which is the actor network.
        self.path = self.conf.vehicle_path + 'SimulationTest/' + agent_name
        self.agent = torch.load(self.path + '/' + agent_name + "_critic.pth")

        print(f"Agent loaded: {run_name}")

        self.map_data = MapData(map_name)


    def sample_poses(self):
        """Sample poses from the map to run through the agent"""
        # n_nvec = 3 
        # n_ctr_pt = 1

        # in the future, make this to sample more poinst along each nvec. 
        # alternatively, sample from the whole map and then assign an orientation to each point that is inside the driveable area
        self.poses = np.zeros((self.map_data.N, 3))
        pts = self.map_data.cline

        diffs = pts[1:] - pts[:-1] 
        angles = np.arctan2(diffs[:,1], diffs[:,0])

        self.poses[:, 0:2] = pts
        self.poses[:-1, 2] = angles

    def plot_poses(self):
        plt.figure(1)
        plt.title(f"Sampled Poses")
        for pose in self.poses:
            plt.arrow(pose[0], pose[1], 0.1*np.cos(pose[2]), 0.1*np.sin(pose[2]), color='r', width=0.01)

        plt.show()


    def find_pose_values(self):
        for pose in self.poses:
            value = self.agent_value_network(pose)
            self.pose_values.append(value)
        
    def plot_values(self):
        pass 




if __name__ == "__main__":
    run_name = "SimulationTest_columbia_small_0_1_0"

    agent = AgentWrapped(run_name)
    agent.sample_poses()
    agent.plot_poses()
