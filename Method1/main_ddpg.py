import numpy as np
from ddpg_torch import Agent
from utils import plot_learning_curve
from fire_model_test import FireEnvironment

import matplotlib.pyplot as plt

if __name__ == '__main__':

    fire_input = "fbndry4.txt"
    dim_x = 30
    dim_y = 30
    dim_z = 5
    agents_theta_degrees = 30
    n_drones = 6
    X_MAX = dim_x - 1
    X_MIN = 0
    Y_MAX = dim_y - 1
    Y_MIN = 0
    Z_MAX = dim_z - 1
    Z_MIN = 1

    n_episodes = 300
    max_steps=1000
    batch_size = 512
    input_dims = 6
    n_actions = 3

    outputs_dir = "results"
    episode_steps = np.zeros(n_episodes).astype(int)
    fov_angle = np.array([30, 30])
    total_reward = 0
    total_rewards = np.zeros(n_episodes).astype(float)

    env = FireEnvironment("fbndry4.txt", dim_x, dim_y, [dim_x, dim_y, dim_z], agents_theta_degrees, n_drones, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, max_steps)

    agents = [Agent(alpha = 0.0001, beta = 0.001, input_dims = input_dims, tau = 0.001,
            batch_size = batch_size, fc1_dims = 400, fc2_dims = 300, n_actions = n_actions) for i in range(n_drones)]

    filename = 'Wildfire_alpha_' + str(agents[0].alpha) + '_beta_' + \
                str(agents[0].beta) + '_' + str(n_episodes) + '_games'
    figure_file = 'plots/' + filename + '.png'

    best_score = env.reward_range[0]

    #Metrics
    score_history = []
    ind_score_history = []
    coverage_ratio_history = []
    coverage_fast_history = np.full(n_episodes, max_steps)
    tracker_fast_history = []
    fire_fallout = []
    env_fallout = []

    for episode in range(n_episodes):
        simulation_index = 0
        observations = np.empty((n_drones,6))
        for i in range(n_drones):
            observations[i] = [np.random.uniform(X_MIN, X_MAX), np.random.uniform(Y_MIN, Y_MAX), np.random.uniform(Z_MIN, Z_MAX), 0, 0, simulation_index]
            agents[i].noise.reset()
        env.reset(observations)
        drones = np.arange(0, n_drones)
        
        #Metrics
        score = 0
        done_history = []
        fire_tracker_status = np.zeros(n_drones)
        fire_fallout_status = np.zeros(n_drones)
        latest_rewards = np.zeros((n_drones))
        prev_on_fire = np.full(n_drones, False)
        env_fallout_status = np.zeros(n_drones)
        while len(drones) > 0:
            env.map = env.simStep(simulation_index)
            simulation_index+=1
            dones = []
            score_tmp = 0
            for i in drones:
                action = agents[i].choose_action(observations[i])
                observation_, reward, done, info = env.step(i, action)
                agents[i].remember(observations[i], action, reward, observation_, done)
                agents[i].learn()
                observations[i] = observation_
                dones.append(done)

                #Metrics
                if done:
                    done_history.append(i)
                    env_fallout_status[i] = info["env_fallout"]
                score_tmp += reward
                latest_rewards[i] += reward
                if fire_tracker_status[i]!=0 and (env.get_drone_view(i)<2).all() and prev_on_fire[i]:
                    fire_fallout_status[i] += 1
                if (env.get_drone_view(i)>1).any():
                    prev_on_fire[i] = True
                    if fire_tracker_status[i]==0:
                        fire_tracker_status[i] = simulation_index
                else:
                    prev_on_fire[i] = False

                env.Render()

            #Metrics
            score += (score_tmp / len(drones))
            drones = drones[np.invert(dones)]
            if np.sum(env.map!=2) & coverage_fast_history[episode]==max_steps:
                coverage_fast_history[episode] = simulation_index
        
        #Metrics
        score_history.append(score/max_steps)
        ind_score_history.append((latest_rewards/max_steps))
        fire_fallout.append(fire_fallout_status)
        env_fallout.append(env_fallout_status)
        avg_score = np.mean(score_history[-1000:])

        if avg_score > best_score:
            best_score = avg_score
            agents[0].save_models()

        print('episode ', episode, ' ', (latest_rewards/max_steps), ' ', 'avg score %.1f' % avg_score)

        #Metrics
        coverage_ratio_history.append((np.sum(env.map==1)/np.sum(env.map>0))*100)
        tracker_fast_history.append(fire_tracker_status)

    x = [i+1 for i in range(n_episodes)]
    plot_learning_curve(x, score_history, figure_file)

    np.savetxt("output_data/score_history.csv", np.array(score_history), delimiter=",", fmt='%.4e')
    np.savetxt("output_data/ind_score_history.csv", np.array(ind_score_history), delimiter=",", fmt='%.4e')
    np.savetxt("output_data/coverage_ratio_history.csv", np.array(coverage_ratio_history), delimiter=",", fmt='%.4e')
    np.savetxt("output_data/coverage_fast_history.csv", coverage_fast_history, delimiter=",", fmt='%.4e')
    np.savetxt("output_data/tracker_fast_history.csv", np.array(tracker_fast_history), delimiter=",", fmt='%.4e')
    np.savetxt("output_data/fire_fallout.csv", np.array(fire_fallout), delimiter=",", fmt='%.4e')
    np.savetxt("output_data/env_fallout.csv", np.array(env_fallout), delimiter=",", fmt='%.4e')
