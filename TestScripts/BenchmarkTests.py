from RacingRewards.f110_gym.f110_env import F110Env
from RacingRewards.Utils.utils import *

from TrainTest import *
from RacingRewards.Planners.PurePursuit import PurePursuit
from RacingRewards.Planners.follow_the_gap import FollowTheGap
from RacingRewards.Planners.AgentPlanner import TrainVehicle, TestVehicle

# MAP_NAME = "porto"
MAP_NAME = "columbia_small"

def pure_pursuit_tests(n=1):
    conf = load_conf("config_file")
    conf.map_name = MAP_NAME

    # runs = zip(['porto', 'columbia_small'], [-0.4, 0])
    runs = zip(['columbia_small'], [0])
    for track, stheta in runs:
        conf.map_name = track
        conf.stheta = stheta
        env = F110Env(map=conf.map_name)

        agent_name = f"PurePursuit_{conf.map_name}_{n}"
        planner = PurePursuit(conf, agent_name)

        eval_dict_wo = evaluate_vehicle(env, planner, conf, True)

        config_dict = vars(conf)
        config_dict['test_number'] = n
        config_dict['eval_name'] = "benchmark"
        config_dict['agent_name'] = agent_name
        config_dict['Wo'] = eval_dict_wo
        config_dict['vehicle'] = "PP"


        save_conf_dict(config_dict)
        env.close_rendering()   


def follow_the_gap_tests(n=1):
    conf = load_conf("config_file")
    conf.map_name = MAP_NAME

    # runs = zip(['porto', 'columbia_small'], [-0.4, 0])
    runs = zip(['columbia_small'], [0])
    for track, stheta in runs:
        conf.map_name = track
        conf.stheta = stheta
        env = F110Env(map=conf.map_name)

        agent_name = f"FGM_{conf.map_name}_{n}"
        planner = FollowTheGap(conf, agent_name)

        eval_dict = evaluate_vehicle(env, planner, conf, True)
        
        config_dict = vars(conf)
        config_dict['test_number'] = n
        config_dict['eval_name'] = "benchmark"
        config_dict['agent_name'] = agent_name
        config_dict['Wo'] = eval_dict
        config_dict['vehicle'] = "FGM"

        save_conf_dict(config_dict)


def benchmark_baseline_tests(n):
    conf = load_conf("config_file")
    conf.rk = 0
    # runs = zip(['porto', 'columbia_small'], [-0.4, 0])
    runs = zip(['columbia_small'], [0])
    for track, stheta in runs:
        conf.map_name = track
        conf.stheta = stheta
        agent_name = f"Baseline_{n}_{track}"
        env = F110Env(map=conf.map_name)

        planner = TrainVehicle(agent_name, conf)
        train_baseline_vehicle(env, planner, conf, False)

        planner = TestVehicle(agent_name, conf)
        eval_dict = evaluate_vehicle(env, planner, conf, True)
        
        config_dict = vars(conf)
        config_dict['test_number'] = n
        config_dict['Wo'] = eval_dict
        config_dict['agent_name'] = agent_name
        config_dict['eval_name'] = "benchmark"
        config_dict['vehicle'] = f"Base{n}"

        save_conf_dict(config_dict)
        env.close_rendering()


if __name__ == "__main__":
    pure_pursuit_tests(1)
    follow_the_gap_tests(1)
    benchmark_baseline_tests(1)