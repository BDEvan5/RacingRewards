import time
import numpy as np
# Test Functions
def evaluate_vehicle(env, vehicle, conf, show=False):
    crashes = 0
    completes = 0
    lap_times = [] 
    laptime = 0.0
    start = time.time()

    for i in range(conf.test_n):
        obs, step_reward, done, info = env.reset()
        while not done:
            action = vehicle.plan(obs)
            sim_steps = conf.sim_steps
            while sim_steps > 0 and not done:
                obs, step_reward, done, _ = env.step(action[None, :])
                sim_steps -= 1

            if show:
                # env.render(mode='human')
                env.render(mode='human_fast')
 
        # env.sim.agents[0].history.plot_history()
        r = find_conclusion(obs, start)

        if r == 1:
            completes += 1
            lap_times.append(env.lap_times[0])
        else:
            crashes += 1

        # env.save_traj(f"Traj_{i}_{vehicle.name}")

    success_rate = (completes / (completes + crashes) * 100)
    if len(lap_times) > 0:
        avg_times, std_dev = np.mean(lap_times), np.std(lap_times)
    else:
        avg_times, std_dev = 0, 0

    print(f"Crashes: {crashes}")
    print(f"Completes: {completes} --> {success_rate:.2f} %")
    print(f"Lap times Avg: {avg_times} --> Std: {std_dev}")

    eval_dict = {}
    eval_dict['success_rate'] = float(success_rate)
    eval_dict['avg_times'] = float(avg_times)
    eval_dict['std_dev'] = float(std_dev)

    print(f"Finished running test and saving file with results.")

    return eval_dict


def train_baseline_vehicle(env, vehicle, conf, show=False):
    start_time = time.time()
    state, step_reward, done, info = env.reset()
    print(f"Starting Baseline Training: {vehicle.name}")
    crash_counter = 0

    ep_steps = 0 
    lap_counter = 0
    for n in range(conf.baseline_train_n + conf.buffer_n):
        state['reward'] = set_reward(state)
        action = vehicle.plan(state)
        sim_steps = conf.sim_steps
        while sim_steps > 0 and not done:
            s_prime, r, done, _ = env.step(action[None, :])
            sim_steps -= 1

        state = s_prime
        if n > conf.buffer_n:
            vehicle.agent.train(2)
        if show:
            env.render('human_fast')
        
        if done or ep_steps > conf.max_steps:
            s_prime['reward'] = set_reward(s_prime) 
            vehicle.done_entry(s_prime)

            print(f"{n}::Lap done {lap_counter} -> FinalR: {s_prime['reward']} -> LapTime {env.lap_times[0]:.2f} -> TotalReward: {vehicle.t_his.rewards[vehicle.t_his.ptr-1]:.2f}")
            lap_counter += 1
            ep_steps = 0 
            if state['reward'] == -1:
                crash_counter += 1

            state, step_reward, done, info = env.reset()
            
        ep_steps += 1

    vehicle.t_his.print_update(True)
    vehicle.t_his.save_csv_data()
    vehicle.agent.save(vehicle.path)

    train_time = time.time() - start_time
    print(f"Finished Training: {vehicle.name} in {train_time} seconds")
    print(f"Crashes: {crash_counter}")

    return train_time, crash_counter



def find_conclusion(s_p, start):
    laptime = s_p['lap_times'][0]
    if s_p['lap_counts'][0] == 1:
        print(f'Complete --> Sim time: {laptime:.2f} Real time: {(time.time()-start):.2f}')
        return 1
    else:
        print(f'Collision --> Sim time: {laptime:.2f} Real time: {(time.time()-start):.2f}')
        return 0



# def find_conclusion(s_p, start):
#     laptime = s_p['lap_times'][0]
#     if s_p['collisions'][0] == 1:
#         print(f'Collision --> Sim time: {laptime:.2f} Real time: {(time.time()-start):.2f}')
#         return -1
#     elif s_p['lap_counts'][0] == 1:
#         print(f'Complete --> Sim time: {laptime:.2f} Real time: {(time.time()-start):.2f}')
#         return 1
#     else:
#         print("No conclusion: Awkward palm trees")
#         # print(s_p)
#     return 0


def set_reward(s_p):
    if s_p['collisions'][0] == 1:
        return -1
    elif s_p['lap_counts'][0] == 1:
        return 1
    return 0

