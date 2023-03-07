import numpy as np
from ddpg_torch import Agent
from utils import plot_learning_curve
from fire_model_test import FireEnvironment
from buffer import ReplayBuffer

import matplotlib.pyplot as plt


def remember(state, action, reward, state_, done):
    memory.store_transition(state, action, reward, state_, done)


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
    max_size = 1000000
    batch_size = 512
    input_dims = 9
    n_actions = 3

    outputs_dir = "results"
    episode_steps = np.zeros(n_episodes).astype(int)
    fov_angle = np.array([30, 30])
    total_reward = 0
    total_rewards = np.zeros(n_episodes).astype(float)

    env = FireEnvironment("fbndry4.txt", dim_x, dim_y, [dim_x, dim_y, dim_z], agents_theta_degrees, n_drones, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, max_steps)

    agents = [Agent(alpha = 0.0001, beta = 0.001, gamma=0.95, input_dims = input_dims, n_actions = n_actions, tau = 0.001,
            batch_size = batch_size, fc1_dims = 400, fc2_dims = 300) for i in range(n_drones)]

    memory = ReplayBuffer(max_size, input_dims, n_actions)

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
        observations = np.zeros((n_drones,9))
        observations_tgt = np.zeros((n_drones,3))

        for i in range(n_drones):
            observations_tgt[i] = [round(np.random.uniform(X_MIN, X_MAX), 4), round(np.random.uniform(Y_MIN, Y_MAX), 4), round(np.random.uniform(Z_MIN, Z_MAX), 4)]
            observations[i] = [round(np.random.uniform(X_MIN, X_MAX), 4), round(np.random.uniform(Y_MIN, Y_MAX), 4), round(np.random.uniform(Z_MIN, Z_MAX), 4), 0, 0, simulation_index, observations_tgt[i][0], observations_tgt[i][1], observations_tgt[i][2]] 
            agents[i].noise.reset()

        env.reset(observations)
        drones = np.arange(0, n_drones)

        for i in range(n_drones):
            map_dict = env.get_drone_view_info(i)
            if 2 in map_dict.keys():
                observations[i][4] = 1.0
            if 3 in map_dict.keys():
                observations[i][3] = 1.0
            map_dict.clear()
        
        #Metrics
        score = 0
        done_history = []
        fire_tracker_status = np.zeros(n_drones)
        fire_fallout_status = np.zeros(n_drones)
        latest_rewards = np.zeros((n_drones))
        prev_on_fire = np.full(n_drones, False)
        env_fallout_status = np.zeros(n_drones)

        avg_alt = np.zeros(n_drones)
        streak = np.zeros(n_drones)
        fire_scores = np.zeros(n_drones)
        
        while len(drones) > 0:
            env.map = env.simStep(simulation_index)
            simulation_index+=1
            dones = []
            score_tmp = 0

            nei_agents = {}
            seeker_net = {}
            nei_agents.clear()
            seeker_net.clear()
            for i in drones:
                nei_agent = []
                for j in range(n_drones):
                    if i!=j and (np.linalg.norm(observations[j][0:3] - observations[i][0:3]))<10:
                        nei_agent.append(j)
                nei_agents[i] = nei_agent
                if (observations[i][3] == 1.0) or (observations[i][4] == 1.0):
                    drone_map = env.get_drone_view(i)
                    fire_points = np.transpose(np.where(drone_map>1))
                    if len(fire_points)>0:
                        tgt_x = round(np.mean(fire_points[:,0]), 4)
                        tgt_y = round(np.mean(fire_points[:,1]), 4)
                        tgt_z = observations[i][2]
                        observations[i][6:9] = [tgt_x, tgt_y, tgt_z]
                        if (np.linalg.norm(np.array([tgt_x, tgt_y, tgt_z]) - observations[i][0:3]))<1:
                            observations[i][6] += round(np.clip(np.random.normal(0, 0.75), -1, 1), 4)   # Noise
                            observations[i][7] += round(np.clip(np.random.normal(0, 0.75), -1, 1), 4)   # Noise

                    streak[i] += 1
                    avg_alt[i] = (avg_alt[i] * (streak[i] - 1) + observations[i][2]) / streak[i]
                    map_dict = env.get_drone_view_info(i)
                    #fire_scores[i] += map_dict[3] if 3 in map_dict.keys() else 0.0
                    if 2 in map_dict.keys():
                        fire_scores[i] += map_dict[2]
                    elif 3 in map_dict.keys():
                        fire_scores[i] += map_dict[3]
                else:
                    seeker_net[i] = nei_agent
                    streak[i] = 0
                    avg_alt[i] = 0
                    fire_scores[i] = 0
            
            follow = {}
            follow.clear()
            for i in seeker_net.keys():
                ranks = np.zeros(len(seeker_net[i]))
                for indj, j in enumerate(seeker_net[i]):
                    if (observations[j][3] == 1.0) or (observations[j][4] == 1.0):
                        map_dict = env.get_drone_view_info(j)
                        curr_fire = 0.0
                        if 2 in map_dict.keys():
                            curr_fire = map_dict[2]
                        if 3 in map_dict.keys():
                            curr_fire = map_dict[3]
                            
                        ranks[indj] = ((fire_scores[j] / (avg_alt[j] * streak[j])) * (curr_fire/observations[j][2]))
                if (ranks>0.0).any():
                    follow[i] = (seeker_net[i][np.argmax(ranks)], np.max(ranks))
                else:
                    follow[i] = (-1, -1)

            # Consensus
            for _ in range(len(seeker_net)):
                for j in seeker_net.keys():
                    for k in seeker_net[j]:
                        if k in seeker_net.keys():
                            if follow[j][1] < follow[k][1]:
                                follow[j] = follow[k]

            for i in drones:
                #print("Drone ",i)
                #print("seeker_net : ", seeker_net)
                #print("follow : ",follow)
                if i in seeker_net.keys() and follow[i][0]!=-1:
                    observations[i][6:9] = observations[follow[i][0]][0:3]
                else:
                    if np.linalg.norm(np.array(observations_tgt[i]) - np.array(observations[i][0:3]))<1:
                        observations_tgt[i] = [round(np.random.uniform(X_MIN, X_MAX), 4), round(np.random.uniform(Y_MIN, Y_MAX), 4), round(np.random.uniform(Z_MIN, Z_MAX), 4)]
                        #print("Reset Target Obs")
                    if observations[i][3] != 1.0:
                        observations[i][6:9] = observations_tgt[i]
                action = agents[i].choose_action(observations[i])
                #print("observations : ",observations)
                #print("action : ",action)
                observation_, reward, done, info = env.step(i, action)
                #print("observation_ : ",observation_)
                remember(observations[i], action, reward, observation_, done)
                if memory.mem_cntr >= batch_size:
                    mem_states, mem_actions, mem_reward, mem_states_, mem_done = \
                        memory.sample_buffer(batch_size)
                    agents[i].learn(mem_states, mem_actions, mem_reward, mem_states_, mem_done)
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

                #if episode > round(n_episodes * 0.9):
                #    env.Render()
                #env.Render()

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
