from RacingRewards.RewardSignals.RewardUtils import *


# Track base
class RaceTrack:
    def __init__(self, map_name) -> None:
        self.wpts = None
        self.ss = None
        self.map_name = map_name
        self.total_s = None

        self.max_distance = 0
        self.distance_allowance = 0.3


    def load_center_pts(self):
        track_data = []
        filename = 'maps/' + self.map_name + '_std.csv'
        
        try:
            with open(filename, 'r') as csvfile:
                csvFile = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  
        
                for lines in csvFile:  
                    track_data.append(lines)
        except FileNotFoundError:
            raise FileNotFoundError("No map file center pts")

        track = np.array(track_data)
        print(f"Track Loaded: {filename} in reward")

        N = len(track)
        self.wpts = track[:, 0:2]
        ss = np.array([get_distance(self.wpts[i], self.wpts[i+1]) for i in range(N-1)])
        ss = np.cumsum(ss)
        self.ss = np.insert(ss, 0, 0)

        self.total_s = self.ss[-1]

        self.diffs = self.wpts[1:,:] - self.wpts[:-1,:]
        self.l2s   = self.diffs[:,0]**2 + self.diffs[:,1]**2 

    def load_reference_pts(self):
        track_data = []
        filename = 'maps/' + self.map_name + '_opti.csv'
        
        try:
            with open(filename, 'r') as csvfile:
                csvFile = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  
        
                for lines in csvFile:  
                    track_data.append(lines)
        except FileNotFoundError:
            raise FileNotFoundError("No reference path")

        track = np.array(track_data)
        print(f"Track Loaded: {filename} in reward")

        self.ss = track[:, 0]
        self.wpts = track[:, 1:3]

        self.total_s = self.ss[-1]

        self.diffs = self.wpts[1:,:] - self.wpts[:-1,:]
        self.l2s   = self.diffs[:,0]**2 + self.diffs[:,1]**2 

    def find_s(self, point):
        dots = np.empty((self.wpts.shape[0]-1, ))
        for i in range(dots.shape[0]):
            dots[i] = np.dot((point - self.wpts[i, :]), self.diffs[i, :])
        t = dots / self.l2s

        t = np.clip(dots / self.l2s, 0.0, 1.0)
        projections = self.wpts[:-1,:] + (t*self.diffs.T).T
        dists = np.linalg.norm(point - projections, axis=1)

        min_dist_segment = np.argmin(dists)
        dist_from_cur_pt = dists[min_dist_segment]
        if dist_from_cur_pt > 1: #more than 2m from centerline
            return self.ss[min_dist_segment] - dist_from_cur_pt # big makes it go back

        s = self.ss[min_dist_segment] + dist_from_cur_pt

        return s 

    def check_done(self, observation):
        position = observation['state'][0:2]
        s = self.find_s(position)

        if s < self.max_distance - self.distance_allowance:
            return -1 # made negative progress
        self.max_distance = min(self.max_distance, s)


def get_base_reward(observation):
    if observation['lap_done']:
        return 1  # complete
    if observation['colision_done']:
        return -1 # crash

    return 0 # intermediate.

class DistanceReward():
    def __init__(self, race_track: RaceTrack) -> None:
        self.race_track = race_track

    def __call__(self, observation, prev_obs):
        if prev_obs is None: return 0

        position = observation['state'][0:2]
        prev_position = prev_obs['state'][0:2]

        s = self.race_track.find_s(position)
        ss = self.race_track.find_s(prev_position)
        reward = (ss - s) / self.race_track.total_s

        reward += get_base_reward(observation)

        return reward
