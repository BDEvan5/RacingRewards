


from pyrsistent import b


class TimeReward:
    def __init__(self, run):
        self.b_time = run.b_time
        self.ref_time = run.ref_time

    def __call__(self, state, s_prime):
        if s_prime['lap_counts'][0] == 1:
            reward = (s_prime['lap_times'][0] - self.ref_time)* self.b_time
            return s_prime['reward'] + reward
        return s_prime['reward']




