from RacingRewards.RewardSignals.RewardUtils import *


# Track base
class TrackPtsBase:
    def __init__(self, config) -> None:
        self.wpts = None
        self.ss = None
        self.map_name = config.map_name
        self.total_s = None

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

    def get_distance_r(self, pt1, pt2, beta):
        s = self.find_s(pt1)
        ss = self.find_s(pt2)
        ds = ss - s
        scale_ds = ds / self.total_s
        r = scale_ds * beta
        shaped_r = np.clip(r, -0.5, 0.5)

        return shaped_r


class DistanceReward(TrackPtsBase):
    def __init__(self, run) -> None:
        TrackPtsBase.__init__(self, run)

        self.load_center_pts()
        # self.load_reference_pts()
        self.b_distance = run.b_distance

    def __call__(self, state, s_prime):
        prime_pos = np.array([s_prime['poses_x'][0], s_prime['poses_y'][0]])
        pos = np.array([state['poses_x'][0], state['poses_y'][0]])
        reward = self.get_distance_r(pos, prime_pos, self.b_distance)

        reward += s_prime['reward']

        return reward


