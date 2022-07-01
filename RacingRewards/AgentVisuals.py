import numpy as np 

from RacingRewards.Utils.utils import *
from RacingRewards.Planners.AgentPlanner import TestVehicle
import torch
from RacingRewards.MapData import MapData
from RacingRewards.f110_gym.laser_models import ScanSimulator2D

map_name = "columbia_small"

class AgentWrapped:
    def __init__(self, agent_name):
        self.agent_name = agent_name
        self.conf = load_conf("config_file")
        run = {'map_name': 'columbia_small'}

        # I need to load the value network in, not the agent which is the actor network.
        self.path = self.conf.vehicle_path + 'SimulationTest/' + agent_name
        self.agent = torch.load(self.path + '/' + agent_name + "_critic.pth")

        print(f"Agent loaded: {agent_name}")

        self.scanner = ScanSimulator2D(20, np.pi)
        self.scanner.set_map("maps/" + map_name + ".yaml", ".png")

        self.map_data = MapData(map_name)
        self.map_data._expand_wpts()

        self.scans = []
        self.values = [] # ordered list of values corressponding to scans

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
    
    def convert_scan_xy(self, scan):
        angles = np.linspace(-np.pi/2, np.pi/2, 20)
        xs = np.sin(angles) * scan 
        ys = np.cos(angles) * scan

        return xs, ys

    def generate_pose_states(self):
        action = np.array([0]) # go straight
        action_tensor = torch.from_numpy(action).float().unsqueeze(0)
        plt.figure(3)
        for pose in self.poses:
            scan = self.scanner.scan(pose, None)

            scan_tensor = torch.from_numpy(scan).float().unsqueeze(0)

            value = self.agent.Q1(scan_tensor, action_tensor).item()
            self.values.append(value)

            if False:
                plt.clf()
                plt.title(f"Scan value: {value}")
                xs, ys = self.convert_scan_xy(scan)
                plt.plot(xs, ys)
                plt.show()

    def plot_poses(self):
        plt.figure(1)
        plt.title(f"Sampled Poses")
        for pose in self.poses:
            plt.arrow(pose[0], pose[1], 0.1*np.cos(pose[2]), 0.1*np.sin(pose[2]), color='r', width=0.01)

        # plt.show()
        plt.pause(0.00001)


    def find_pose_values(self):
        action = np.array([0]) # go straight
        action_tensor = torch.from_numpy(action).float().unsqueeze(0)
        for scan in self.scans:
            scan_tensor = torch.from_numpy(scan).float().unsqueeze(0)

            value = self.agent.Q1(scan_tensor, action_tensor)
            self.values.append(value.item())

        print(f"Scan values: {self.values}")
        
    def plot_values(self):
        plt.figure(2)
        plt.title(f"Pose Values")
        for i, value in enumerate(self.values):
            # plt.text(self.poses[i, 0], self.poses[i, 1], f"{value:.2f}")
            pose = self.poses[i]
            # plt.text(pose[0], pose[1], f"{value:.2f}")
            plt.plot(pose[0], pose[1], 'o', color='r', markersize=value)
            print(f"x, {pose[0]} :: y, {pose[1]} :: value, {value}")
            # plt.arrow(pose[0], pose[1], 0.1*np.cos(pose[2]), 0.1*np.sin(pose[2]), color='r', width=0.01)

        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlim([-5, 12])
        plt.ylim([-7, 5])
        
        plt.show()




if __name__ == "__main__":
    run_name = "SimulationTest_columbia_small_0_2_0"

    agent = AgentWrapped(run_name)
    agent.sample_poses()
    agent.generate_pose_states()
    agent.find_pose_values()
    # agent.plot_poses()
    agent.plot_values()
