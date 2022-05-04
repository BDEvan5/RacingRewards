

class BaselineReward:
    def __init__(self, run) -> None:
        self.r_bias = run.r_bias

    def __call__(self, state, s_prime):
        return self.r_bias + s_prime['reward']
