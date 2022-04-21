from RacingRewards.f110_gym.f110_env import F110Env
from RacingRewards.Utils.utils import *

from TrainTest import *
from RacingRewards.Planners.AgentPlanner import TrainVehicle, TestVehicle
from RacingRewards.RewardSignals.CrossTrackReward import CrossTrackReward



def base_test():
    conf = load_conf("config_file")
    runs = setup_run_list("DebugRuns")

    for run in runs:
        env = F110Env(map=run.map_name)

        planner = TrainVehicle(run, conf)
        planner.calculate_reward = CrossTrackReward(run)
        train_baseline_vehicle(env, planner, conf, True)

        planner = TestVehicle(run, conf)
        eval_dict = evaluate_vehicle(env, planner, conf, True)

           
        run_dict = vars(run)
        run_dict.update(eval_dict)

        save_conf_dict(run_dict)
        env.close_rendering()     



if __name__ == "__main__":
    base_test()