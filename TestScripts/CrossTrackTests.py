from RacingRewards.f110_gym.f110_env import F110Env
from RacingRewards.Utils.utils import *

from TrainTest import *
from RacingRewards.Planners.AgentPlanner import TrainVehicle, TestVehicle
from RacingRewards.RewardSignals.CrossTrackReward import CrossTrackReward



def crosstrack_base_test():
    conf = load_conf("config_file")
    runs = setup_run_list("CrossTrackRuns")

    for run in runs:
        env = F110Env(map=run.map_name)

        planner = TrainVehicle(run.run_name, conf)
        planner.calculate_reward = CrossTrackReward(run)
        train_baseline_vehicle(env, planner, conf, False)

        planner = TestVehicle(run.run_name, conf)
        eval_dict = evaluate_vehicle(env, planner, conf, True)

           
        config_dict = vars(conf)
        config_dict['run'] = vars(run)
        config_dict['Wo'] = eval_dict

        save_conf_dict(config_dict)
        env.close_rendering()     



if __name__ == "__main__":
    crosstrack_base_test()