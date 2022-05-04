from RacingRewards.f110_gym.f110_env import F110Env
from RacingRewards.Utils.utils import *

from TrainTest import *
from RacingRewards.Planners.AgentPlanner import TrainVehicle, TestVehicle
from RacingRewards.RewardSignals.BaselineReward import BaselineReward



def baseline_tests():
    conf = load_conf("config_file")
    runs = setup_run_list("BaselineRuns")

    for run in runs:
        env = F110Env(map=run.map_name)

        planner = TrainVehicle(run, conf)
        planner.calculate_reward = BaselineReward(run)
        train_baseline_vehicle(env, planner, conf, False)

        planner = TestVehicle(run, conf)
        eval_dict = evaluate_vehicle(env, planner, conf, False)

           
        # config_dict = vars(conf)
        run_dict = vars(run)
        run_dict.update(eval_dict)

        save_conf_dict(run_dict)
        env.close_rendering()     



if __name__ == "__main__":
    baseline_tests()