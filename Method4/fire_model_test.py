#adapted and copied from https://github.com/eczy/rl-drone-coverage/blob/master/field_coverage_env.py

import gym
import numpy as np
import cv2
import copy
import math

class FireEnvironment(gym.Env):

    class FireInfo(object):
        def __init__(self, x, y):
            self.x = x
            self.y = y

    class Agent(object):
        def __init__(self, pos, fov, fsen, tstep, target):
            self.pos = pos
            self.fov = fov
            self.fsen = fsen
            self.tstep = tstep
            self.target = target

        def print_agent(self):
            out = []
            for i in range(len(self.pos)):
                out.append(self.pos[i])
            for i in range(len(self.fsen)):
                out.append(self.fsen[i])
            out.append(self.tstep)
            for i in range(len(self.target)):
                out.append(self.target[i])
            return out

    def __init__ (self, data_file_name, height, width, shape, theta, num_agents,  X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, max_steps):
        super().__init__()

        self.fire_data = self.readFireData(data_file_name)

        self.height = height
        self.width = width
        self.shape = shape
        self.theta = np.radians(theta)
        self.num_agents = num_agents
        self.X_MIN = X_MIN
        self.X_MAX = X_MAX
        self.Y_MIN = Y_MIN
        self.Y_MAX = Y_MAX
        self.Z_MIN = Z_MIN
        self.Z_MAX = Z_MAX
        self.max_steps = max_steps
        self.steps = 0
        #self.kick_start = 7500
        self.kick_start = 6000

        self.action_space = gym.spaces.Box(-1, +1, (3,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=np.array(shape))

        self.agents = {}
        self.env_fallout = np.zeros(self.num_agents)

    def readFireData(self, file_name):
        f = open(file_name, "r")
        return_data = []
        while True:
            data = f.readline()
            if not data:
                break
            vals = data.split(",")
            if(len(vals) > 1):
                return_data.append(vals)

        return return_data


    def simStep(self, time_t):
        fire_map = {}
        vals = self.fire_data[time_t + self.kick_start]
        self.steps = time_t
        #vals = self.fire_data[900]
        self.fire_info = []
        for j in range(0,len(vals)-1,3):
            x = int(vals[j]) - (self.X_MAX/3)
            y = int(vals[j+1]) - (self.Y_MAX/3)

            if str(str(x) + "," + str(y)) in fire_map:
                pass
            else:
               fire_map[str(x) + "," + str(y)] = True
               temp = self.FireInfo(int(x), int(y))
               self.fire_info.append(temp)

        for t in self.fire_info:
            if self.map[t.x, t.y] < 1:
                self.map[t.x, (t.y-1 if t.y>self.Y_MIN else self.Y_MIN):(t.y+2 if t.y<self.Y_MAX else self.Y_MAX-1)] = 2
                self.map[(t.x-1 if t.x>self.X_MIN else self.X_MIN):(t.x+2 if t.x<self.X_MAX else self.X_MAX-1), t.y] = 2
                
        fire_map.clear()
        return self.map


    def reset(self, observations):
        self.steps = 0
        agents = {}
        self.map = np.zeros((self.height,self.width), np.uint8)
        self.map = self.simStep(0)
        self.agents.clear()
        for i in range(self.num_agents):
            agents[i] = self.Agent(observations[i][0:3], self.theta, observations[i][3:5], observations[i][5], observations[i][6:9])
        self.agents = agents

        self.env_fallout = np.zeros(self.num_agents)

    def state(self, drone):
        return self.agents[drone].print_agent()


    def step(self, drone, action):
        done = self.steps >= self.max_steps
        success_text = {'STATUS': done}

        prev_pos = self.agents[drone]
        self.move_drone(drone, action)

        mask = self.view_ind_mask(drone)
        foi = self.map.astype(int)

        #for map, view in zip(np.rot90(foi), np.rot90(self.get_drone_view(drone))):
        #    print( map, "  ", view )

        rewards = 0
        rew_list = {}

        # Reward Based on Discovery:
        map_dict = self.get_drone_view_info(drone)
        if 1 in map_dict.keys():
            #rewards += ( map_dict[1] * -10)
            rewards += ( map_dict[1] * -4)

        if 2 in map_dict.keys():
            #rewards += ( map_dict[2] * -4 )
            rewards += ( map_dict[2] * 6 )
            #obs_seen_fire = 1.0
            self.agents[drone].fsen[1] = 1.0
        else:
            #obs_seen_fire = 0.0
            self.agents[drone].fsen[1] = 0.0

        if 3 in map_dict.keys():
            #rewards += ( map_dict[3] * 1 )
            rewards += ( map_dict[3] * 12 )
            #obs_seen_new_fire = 1.0
            self.agents[drone].fsen[0] = 1.0
        else:
            #obs_seen_new_fire = 0.0
            self.agents[drone].fsen[0] = 0.0
        rew_list[1] = rewards
        '''
        if self.agents[drone].fsen[0] == 1.0:
            drone_map = self.get_drone_view(drone)
            fire_points = np.transpose(np.where(drone_map > 2))
            self.agents[drone].target = [ round(np.mean(fire_points[:,0]), 4), round(np.mean(fire_points[:,1]), 4), 1]
        else:
            nei_euc_dist = {}
            for i in nei_agents:
                if self.agents[i].fsen[0] == 1.0:
                    nei_euc_dist[i] = np.abs(np.linalg.norm(np.array(self.agents[drone].pos) - np.array(self.agents[i].pos)))

            if len(nei_euc_dist) > 0:
                min_idx = min(nei_euc_dist, key=nei_euc_dist.get)
                self.agents[drone].target = self.agents[min_idx].pos
                rewards += -(nei_euc_dist[min_idx]**2)
            else:
                self.agents[drone].target = [-1, -1, -1]
        '''

        if (self.agents[drone].target!=-1).all():
            #print("\nagent : ",self.agents[drone].print_agent())
            target_angle = np.abs( round( math.degrees( math.atan2(self.agents[drone].target[1] - prev_pos.pos[1], self.agents[drone].target[0] - prev_pos.pos[0])), 4) )
            pred_angle = np.abs( round( math.degrees( math.atan2(self.agents[drone].pos[1] - prev_pos.pos[1], self.agents[drone].pos[0] - prev_pos.pos[0])), 4) )
            angle_diff = round(np.abs(target_angle - pred_angle), 4)
            #print("angle_diff : ",angle_diff)
            #gain = round( 20 - np.log( angle_diff if angle_diff>=1.0 else 1.0 ) * 4 )
            gain = round( ( 0.94 ** angle_diff ) * 20, 4 )
            rewards += gain
            rew_list[2] = gain

        #print(self.agents[drone].print_agent())

        # Marking the Undiscovered Fire Points as Discovered:
        disc_map = ( foi - ( mask * 3 ) ).astype(int)
        disc_map[disc_map==-3] = 0
        disc_map[disc_map<0] = 1
        self.map = disc_map
        
        mod_pos = prev_pos.pos + action

        #rewards += -((self.Z_MAX+2 - mod_pos[2]) * 10)
        #rewards += (mod_pos[2] * 2)

        if (self.X_MIN > mod_pos[0]) or (mod_pos[0] > self.X_MAX) or (self.Y_MIN > mod_pos[1]) or (mod_pos[1] > self.Y_MAX):
            rewards = -150
            rew_list[3] = -150
            #done = True
            #success_text = {'STATUS': done}
            self.env_fallout[drone] += 1
        
        self.agents[drone].tstep += 1
        observation_ = self.state(drone)

        if done:
            success_text["env_fallout"] = self.env_fallout[drone]
        #print("rewards : ",rew_list)
        return observation_, np.round(rewards), done, success_text

    def get_drone_view(self, drone):
        mask = self.view_ind_mask(drone)
        foi = self.map.astype(int)

        return ( ( foi + mask ) * mask ).astype(int)

    def get_drone_view_info(self, drone):
        map = self.get_drone_view(drone)
        field_type, counts = np.unique(map, return_counts=True)
        map_dict = dict(zip(field_type, counts))

        return map_dict

    def view_ind_mask(self, drone):
        coordsx, coordsy = np.meshgrid(*[np.arange(x) for x in self.map.shape])
        mask = np.zeros_like(self.map).astype(int)
        x, y, z = self.agents[drone].pos

        for xc, yc in zip(coordsx.flatten(), coordsy.flatten()):
            x_proj = y_proj = np.tan(self.agents[drone].fov) * z
            if all([
                xc > x - x_proj,
                xc < x + x_proj,
                yc > y - y_proj,
                yc < y + y_proj
            ]):
                mask[xc, yc] = True
        return mask



    def move_drone(self, drone, action):
        X, Y, Z = self.shape
        x, y, z = self.agents[drone].pos

        action = (action / 10)

        new_pos = [round(min(max(x + action[0], self.X_MIN), self.X_MAX), 4), round(min(max(y + action[1], self.Y_MIN), self.Y_MAX), 4), round(min(max(z + action[2], self.X_MIN + 1), self.Z_MAX), 4)]
        self.agents[drone].pos = new_pos

    def Render(self):
        temp_img = np.zeros((self.height,self.width), np.uint8)
        for t in self.fire_info:
            temp_img = cv2.circle(temp_img, (t.x, t.y), 1, (100,100,100), -1)
        temp_img = cv2.normalize(temp_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        color = (255, 0, 0)
        thickness = -1
        temp = cv2.cvtColor(temp_img, cv2.COLOR_GRAY2BGR)

        for drone in range(self.num_agents):
            fov_alt = round(np.tan(self.theta) * self.agents[drone].pos[2])
            start_point = (round(self.agents[drone].pos[0])-fov_alt, round(self.agents[drone].pos[1])-fov_alt)
            end_point = (round(self.agents[drone].pos[0])+fov_alt, round(self.agents[drone].pos[1])+fov_alt)
            temp = cv2.rectangle(temp, start_point, end_point, (0, 1, 1), -1)
            temp = cv2.circle(temp, (round(self.agents[drone].pos[0]), round(self.agents[drone].pos[1])), 0, color, thickness)

        cv2.namedWindow("simulation", cv2.WINDOW_NORMAL)
        temp = cv2.resize(temp, (1000,1000), interpolation = cv2.INTER_AREA)
        cv2.imshow("simulation", temp)
        cv2.waitKey(1)

    def render(self):
        temp_img = self.map.copy()
        color = (255, 0, 0)
        thickness = -1
        temp = cv2.cvtColor(temp_img, cv2.COLOR_GRAY2BGR)
        for drone in range(len(self.agents)):
            temp = cv2.circle(temp, (round(self.agents[drone].pos[0]), round(self.agents[drone].pos[1])), 1, color, thickness)

        temp = cv2.resize(temp, (220,220), interpolation = cv2.INTER_AREA)
        cv2.imshow("filled", temp)
        cv2.waitKey(1)


    def print(self, file):
        f =  open(file, 'a+')
        coordsx, coordsy = np.meshgrid(*[np.arange(x) for x in self.map.shape])
        view_masks = {}
        for i, drone in self.agents.items():
            mask = np.zeros_like(self.map).astype(int)
            x, y, z = drone.pos
            x_proj = 0
            y_proj = 0
            for xc, yc in zip(coordsx.flatten(), coordsy.flatten()):
                x_proj = np.tan(drone.fov) * z
                y_proj = np.tan(drone.fov) * z

            #print(i, x, y, z, x_proj, y_proj)
            f.write(str(str(i) + " " + str(x) + " " + str(y) + " " + str(z) + " " + str(x_proj) + " " + str(y_proj)) + "\n")
        f.close()


def main():
    n_drones = 3
    dim_x = 15
    dim_y = 15
    dim_z = 3
    agents_theta_degrees = 30
    X_MAX = 14
    X_MIN = 0
    Y_MAX = 14
    Y_MIN = 0
    Z_MAX = 4
    Z_MIN = 0     # z min = 1

    env = FireEnvironment("fbndry4.txt", dim_x, dim_y, [dim_x, dim_y, dim_z], agents_theta_degrees, n_drones, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX)
    for i in range(200):
        env.grid = env.simStep(i)
        env.map = env.grid
        env.render()
        env.print(("sim.txt"))
    print(env.observation_space.shape)
    print(env.action_space.shape)

if __name__ == "__main__":
    main()
