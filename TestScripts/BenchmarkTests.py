from RacingRewards.f110_gym.f110_env import F110Env
from RacingRewards.Utils.utils import *

from TrainTest import *
from RacingRewards.Planners.PurePursuit import PurePursuit
from RacingRewards.Planners.follow_the_gap import FollowTheGap


def purepursuit_base_test():
    conf = load_conf("config_file")
    runs = setup_run_list("BenchmarkRuns")

    for run in runs:
        run.reward_name = "PP"
        run.path = "PP/"
        env = F110Env(map=run.map_name)

        planner = PurePursuit(conf, run)
        eval_dict = evaluate_vehicle(env, planner, conf, True)
        # eval_dict = evaluate_vehicle(env, planner, conf, False)

        run_dict = vars(run)
        run_dict.update(eval_dict)

        save_conf_dict(run_dict)
        env.close_rendering()     


def fgm_base_test():
    conf = load_conf("config_file")
    runs = setup_run_list("BenchmarkRuns")

    for run in runs:
        run.reward_name = "FGM"
        run.path = "FGM/"
        env = F110Env(map=run.map_name)

        planner = FollowTheGap(conf, run)
        eval_dict = evaluate_vehicle(env, planner, conf, False)

        run_dict = vars(run)
        run_dict.update(eval_dict)

        save_conf_dict(run_dict)
        env.close_rendering()     





if __name__ == "__main__":

    purepursuit_base_test()
    # fgm_base_test()