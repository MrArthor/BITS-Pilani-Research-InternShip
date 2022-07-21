
import gym
import sys
import numpy as np
from decimal import Decimal
from my_utils import *
from load_and_save_data import *
import scenario_objects
import agent
import plotting
from gym import spaces, logger
import os

load = Loader()
load.maps_data()
load.users_clusters()
load.maps_status()
plot = plotting.Plot()

class UAVEnv(gym.Env):

    def __init__(self):

        upper_limits = np.array([val[0] for val in LIMIT_VALUES_FOR_ACTION])
        lower_limits = np.array([val[1] for val in LIMIT_VALUES_FOR_ACTION])

        self.action_set_min = ACTION_SPACE_3D_MIN if DIMENSION_2D==False else ACTION_SPACE_2D_MIN
        if (UNLIMITED_BATTERY==True):
            self.q_table_action_set = self.action_set_min
        else:
            if (self.action_set_min==ACTION_SPACE_2D_MIN):
                self.q_table_action_set = ACTION_SPACE_2D_TOTAL
                self.charging_set = ACTION_SPACE_2D_WHILE_CHARGING
                self.come_home_set = ACTION_SPACE_2D_COME_HOME
            else:
                self.q_table_action_set = ACTION_SPACE_3D_TOTAL
                self.charging_set = ACTION_SPACE_3D_WHILE_CHARGING
                self.come_home_set = ACTION_SPACE_3D_COME_HOME
        self.action_space = spaces.Discrete(len(self.q_table_action_set))
        self.nb_actions = self.action_space.n
        self.observation_space = spaces.Box(low=0, high=max(CELLS_ROWS, CELLS_COLS, FULL_BATTERY_LEVEL), shape=(CELLS_ROWS, CELLS_COLS, ITERATIONS_PER_EPISODE), dtype=np.float32) if DIMENSION_2D==False else spaces.Box(low=0, high=max(CELLS_ROWS, CELLS_COLS), shape=(CELLS_ROWS, CELLS_COLS), dtype=np.float32)
        self.state = None
        self.obs_points = load.obs_points
        self.points_matrix = load._points_matrix
        self.cs_points = load.cs_points
        self.eNB_point = load.enb_point
        self.cells_matrix = load.cells_matrix
        self.obs_cells = load.obs_cells
        self.max_uav_height = max([obs._z_coord for obs in self.obs_cells]) if DIMENSION_2D==False else 0
        self.cs_cells = load.cs_cells

        self.initial_uavs_pos = agent.Agent.setting_agents_pos(self.cs_cells) if UNLIMITED_BATTERY==False else UAVS_POS
        self.agents = agent.Agent.initialize_agents(self.initial_uavs_pos, self.max_uav_height, self.action_set_min) 
        self.eNB_cells = load.enb_cells
        self.points_status_matrix = load.points_status_matrix
        self.cells_status_matrix = load.cells_status_matrix
        self.clusterer = load.initial_clusterer
        self.cluster_centroids = load.initial_centroids
        self.users_clusters = load.initial_usr_clusters
        self.clusters_radiuses = load.initial_clusters_radiuses
        initial_users = load.initial_users
        for user in initial_users:
            user._x_coord /= CELL_RESOLUTION_PER_COL 
            user._y_coord /= CELL_RESOLUTION_PER_ROW
        self.users = initial_users
        self.users_walk_steps = []
        self.k_steps_to_walk = 0
        self.uav_footprint = ACTUAL_UAV_FOOTPRINT 
        self.n_users = len(self.users)
        self.discovered_users = []
        self.current_requested_bandwidth = 0 
        self.all_users_in_all_foots = []
        self.n_active_users = 0
        self.n_tr_active = 0
        self.n_ec_active = 0
        self.n_dg_active = 0
        self.agents_paths = [[self.get_agent_pos(self.agents[uav])] for uav in range(N_UAVS)]
        self.last_render = 0
        self.instant_to_render = 0

    def step_agent(self, agent, action):
        
        info = ""


        self.agents_paths[agent._uav_ID].append(self.get_agent_pos(agent))
     
        if (UAV_STANDARD_BEHAVIOUR==False):
            current_action = self.q_table_action_set[action]
            agent_pos_ = agent.move(current_action, self.cells_matrix)
        else:
            current_action = ACTION_SPACE_STANDARD_BEHAVIOUR.index(action)
            agent_pos_ = agent.move_standard_behaviour(action)

        if ( ((action==CHARGE_2D_INDEX) or (action==CHARGE)) or ((action==GO_TO_CS_3D_INDEX) or (action==CHARGE)) ):
            agent._users_in_footprint = []
            current_users_in_footprint = []
        else:
            self.current_requested_bandwidth = 0
            current_users_in_footprint, bandwidth_request_in_current_footprint = agent.users_in_uav_footprint(self.users, self.uav_footprint, self.discovered_users)
            if (MULTI_SERVICE==False):
                agent._users_in_footprint = current_users_in_footprint
            else:
                self.current_requested_bandwidth = bandwidth_request_in_current_footprint

        n_served_users = agent.n_served_users_in_foot(agent._users_in_footprint)
        for user_per_agent_foot in current_users_in_footprint:
            self.all_users_in_all_foots.append(user_per_agent_foot) 
        agent._x_coord = agent_pos_[0]
        agent._y_coord = agent_pos_[1]
        if (DIMENSION_2D==False):
            agent._z_coord = agent_pos_[2]

        if (UNLIMITED_BATTERY==True):
            reward = self.reward_function_1(agent._users_in_footprint)
            s_ = (agent_pos_)
        else:
            if (MULTI_SERVICE==False):
                reward = self.reward_function_2(agent._users_in_footprint, agent._battery_level, agent._required_battery_to_CS)
            else:
                if ( (MULTI_SERVICE==True) and (INF_REQUEST==False) ):
                    reward = self.reward_function_3(agent._users_in_footprint, agent._battery_level, agent._required_battery_to_CS, self.n_tr_active, self.n_ec_active, self.n_dg_active)
            s_ = (agent_pos_, agent._battery_level)

        done, info = self.is_terminal_state(agent)

        if (done):
            
            if (info=="IS CRASHED"):
                reward = 0.0
            else:
                reward = 0.0


        if (agent._uav_ID==0):
            self.last_render += 1

        return s_, reward, done, info

    def step(self, actions):

        obs = [0 for uav in range(N_UAVS)]
        rewards = [0 for uav in range(N_UAVS)]
        dones = [0 for uav in range(N_UAVS)]
        infos = [0 for uav in range(N_UAVS)]

        for uav in range(N_UAVS):

            ob, r, d, i = self.step_agent(self.agents[uav], actions[uav])
            obs[uav] = ob
            rewards[uav] = r
            dones[uav] = d
            infos[uav] = i


        return obs, rewards, dones, infos

    def cost_reward(self, battery_level, needed_battery):

        alpha_s = 0
        alpha_c = 0

        if (battery_level > CRITICAL_BATTERY_LEVEL):
            alpha_s = 1
            alpha_c = 0
        elif ( (battery_level >= CRITICAL_BATTERY_LEVEL_2) and (battery_level > needed_battery) ):
            alpha_s = 0.8
            alpha_c = 0.2
        elif ( (battery_level >= CRITICAL_BATTERY_LEVEL_3) and (battery_level > needed_battery) ):
            alpha_s = 0.5
            alpha_c = 0.5
        elif ( (battery_level >= CRITICAL_BATTERY_LEVEL_4) and (battery_level > needed_battery) ):
            alpha_s = 0.2
            alpha_c = 0.8
        elif (battery_level <= needed_battery):
            alpha_s = 0
            alpha_c = 1
            
        reward_for_cost = needed_battery/battery_level if battery_level != 0 else 1

        return reward_for_cost, alpha_s, alpha_c

    def discount_for_user_wait(self):

        n_discovered_users = len(self.discovered_users)
        all_wait_times = sum([wait_time_for_cur_user._info[3] for wait_time_for_cur_user in self.discovered_users])
        avg_wait_time_for_disc_users = all_wait_times/n_discovered_users if n_discovered_users!=0 else 0.0

        discount_factor = 0.0
        if (avg_wait_time_for_disc_users>CRITICAL_WAITING_TIME_FOR_SERVICE):
            discount_factor = CRITICAL_WAITING_TIME_FOR_SERVICE/NORMALIZATION_FACTOR_WAITING_TIME_FOR_SERVIE if avg_wait_time_for_disc_users<=NORMALIZATION_FACTOR_WAITING_TIME_FOR_SERVIE else 1.0
        else:
            discount_factor = 0.0

        return discount_factor

    def reward_function_1(self, users_in_footprint):

        n_users_in_footprint = len(users_in_footprint)
        reward = n_users_in_footprint/(self.n_users/N_UAVS)
        
        if (reward>1):
            reward = 1.0
        
        discount_for_wait_time = self.discount_for_user_wait()
        reward -= discount_for_wait_time
        if (reward<0.0):
            reward = 0.0

        return reward

    def reward_function_2(self, users_in_footprint, battery_level, needed_battery):

        reward_for_users = self.reward_function_1(users_in_footprint)

        alpha_s = 1
        alpha_c = 0
        if (needed_battery==None):
            
            reward = alpha_s*reward_for_users

        else:

            reward_for_cost, alpha_s, alpha_c = self.cost_reward(battery_level, needed_battery)
            reward = alpha_s*reward_for_users + alpha_c*reward_for_cost

        return reward

    def reward_function_3(self, users_in_footprint, battery_level, needed_battery, n_tr_active_users, n_ec_active_users, n_dg_active_users):

        reward_for_users = self.reward_function_1(users_in_footprint)

        served_users_asking_for_service = 0
        n_served_tr_users = 0
        n_served_ec_users = 0
        n_served_dg_users = 0
        for user in users_in_footprint:
            
            if (user._info[1]!=NO_SERVICE):
                
                if (user._info[1]==THROUGHPUT_REQUEST):
                    n_served_tr_users += 1
                elif (user._info[1]==EDGE_COMPUTING):
                    n_served_ec_users += 1
                elif (user._info[1]==DATA_GATHERING):
                    n_served_dg_users += 1
                
                served_users_asking_for_service += 1

        n_tr_served_perc = n_served_tr_users/n_tr_active_users if n_tr_active_users!=0 else 0
        n_ec_served_perc = n_served_ec_users/n_ec_active_users if n_ec_active_users!=0 else 0
        n_dg_served_perc = n_served_dg_users/n_dg_active_users if n_dg_active_users!=0 else 0

        alpha_u = 0.5
        alpha_tr = 0.2
        alpha_ec = 0.2
        alpha_dg = 0.1

        reward_for_services_and_users = alpha_u*reward_for_users + alpha_tr*n_tr_served_perc + alpha_ec*n_ec_served_perc + alpha_dg*n_dg_served_perc
        reward = reward_for_services_and_users

        if (needed_battery!=None):

            reward_for_cost, alpha_s, alpha_c = self.cost_reward(battery_level, needed_battery)
            reward = alpha_s*reward_for_services_and_users + alpha_c*reward_for_cost

        discount_for_wait_time = self.discount_for_user_wait()
        reward -= discount_for_wait_time
        if (reward<0.0):
            reward = 0.0
        
        return reward

    def get_active_users(self):
        n_active_users = 0
        n_inactive_users = 0 
        tr_users = 0
        ec_users = 0
        dg_users = 0

        n_tr_served = 0
        n_ec_served = 0
        n_dg_served = 0        

        for user in self.users:
            if (user._info[1]==NO_SERVICE):
                n_inactive_users += 1
            elif (user._info[1]==THROUGHPUT_REQUEST):
                tr_users += 1
                if (user._info[0]==True):
                    n_tr_served += 1
            elif (user._info[1]==EDGE_COMPUTING):
                ec_users += 1
                if (user._info[0]==True):
                    n_ec_served += 1
            elif (user._info[1]==DATA_GATHERING):
                dg_users += 1
                if (user._info[0]==True):
                    n_dg_served += 1

        n_active_users = tr_users + ec_users + dg_users
        
        self.n_active_users = n_active_users
        self.n_tr_active = tr_users
        self.n_ec_active = ec_users
        self.n_dg_active = dg_users

        return n_active_users, tr_users, ec_users, dg_users, n_tr_served, n_ec_served, n_dg_served

    def set_action_set(self, agent):

        if ( (agent._battery_level <= CRITICAL_BATTERY_LEVEL) and (agent._charging == False) ):
            agent._action_set = ACTION_SPACE_2D_COME_HOME if DIMENSION_2D==True else ACTION_SPACE_3D_COME_HOME 
        elif (agent._charging == True):
            agent._action_set = ACTION_SPACE_2D_WHILE_CHARGING if DIMENSION_2D==True else ACTION_SPACE_3D_WHILE_CHARGING
        elif ( (agent._coming_home == False) and (agent._charging == False) and (agent._battery_level > CRITICAL_BATTERY_LEVEL) ):
            agent._action_set = ACTION_SPACE_2D_MIN if DIMENSION_2D==True else ACTION_SPACE_3D_MIN
            agent._path_to_the_closest_CS = []
            agent._current_pos_in_path_to_CS = -1
            agent._required_battery_to_CS = None

        self.action_space = spaces.Discrete(len(agent._action_set))

    def noisy_measure_or_not(self, values_to_warp):

        warped_values = []
        coord_idx = 1

        for value in values_to_warp:
            noise_prob = np.random.rand()
            
            if noise_prob < 0.1:
                gaussian_noise = np.random.normal(loc=0, scale=1)
                warped_value = round(value + gaussian_noise)
                
               
                if (coord_idx==1):
                    if (warped_value>=AREA_WIDTH):
                        warped_value = AREA_WIDTH - 0.5
                    elif (warped_value<=0):
                        warped_value = 0.5
                    else:
                        warped_value += 0.5
                
              
                elif (coord_idx==2):
                    if (warped_value>=AREA_HEIGHT):
                        warped_value = AREA_HEIGHT - 0.5
                    elif (warped_value<=0):
                        warped_value = 0.5
                    else:
                        warped_value += 0.5
                
               
                elif (coord_idx==3):
                    warped_value = value 
            else:
                warped_value = value

            coord_idx += 1

            warped_values.append(warped_value)
        
        return tuple(warped_values)

    def is_terminal_state(self, agent):

        if (UNLIMITED_BATTERY==False):
            if ( (agent._battery_level <= 0) and (not agent.check_if_on_CS()) ):
                agent._battery_level = 0
                agent._crashed = True

                return True, "IS CRASHED"

            elif (agent._charging==True):
                agent._crashed = False

                return True, "IS CHARGING"

            else:
                agent._crashed = False
                
                return False, "IS WORKING"
        else:
            return False, "IS WORKING" 
    def get_agent_pos(self, agent):

        x = agent._x_coord
        y = agent._y_coord
        z = agent._z_coord

        return (x, y) if DIMENSION_2D==True else (x, y, z)

    def render(self, where_to_save=None, episode=None, how_often_render=None):
        
        self.instant_to_render += 1

        if ( (self.instant_to_render==how_often_render) or (how_often_render==None) ):
            
            if (where_to_save!=None):
                print("\nSaving animation for episode:", episode)

            plot.plt_map_views(obs_cells=self.obs_cells, cs_cells=self.cs_cells, enb_cells=self.eNB_cells,
                               points_status_matrix=self.points_status_matrix, cells_status_matrix=self.cells_status_matrix, users=self.users,
                               centroids=self.cluster_centroids, clusters_radiuses=self.clusters_radiuses, area_height=AREA_HEIGHT,
                               area_width=AREA_WIDTH, N_cells_row=CELLS_ROWS, N_cells_col=CELLS_COLS, agents_paths=self.agents_paths,
                               path_animation=True, where_to_save=where_to_save, episode=episode, last_render=self.last_render)

            self.instant_to_render = 0

        self.last_render = 0
        self.agents_paths = [[] for uav in range(N_UAVS)]


    def reset_uavs(self, agent,):

        if (agent._battery_level == 0):
            agent._battery_level = FULL_BATTERY_LEVEL
            arise_pos_idx = np.random.choice(range(N_UAVS))
            arise_pos = self.initial_uavs_pos[arise_pos_idx]
            agent._x_coord = arise_pos[0]
            agent._y_coord = arise_pos[1]
            agent._z_coord = arise_pos[2]

            agent._charging = False
            agent._coming_home = False
    def reset(self):
        for uav in range(N_UAVS):  
            self.reset_uavs(self.agents[uav])
        # obs = [0 for uav in range(N_UAVS)]
        # return obs
    def update_users_requests(self, users):

        for user in users:
            if (user._info[1]==NO_SERVICE):
                
                type_of_service = scenario_objects.User.which_service()

                if (type_of_service == THROUGHPUT_REQUEST):
                    service_quantity = scenario_objects.User.bitrate_request()
                elif (type_of_service == EDGE_COMPUTING):
                    service_quantity = scenario_objects.User.edge_computing_request()
                elif (type_of_service == DATA_GATHERING):
                    service_quantity = scenario_objects.User.data_gathering()
                else:
                    service_quantity = 0

                requested_service_life = scenario_objects.User.needed_service_life(type_of_service) if type_of_service!=NO_SERVICE else 0

                user._info[1] = type_of_service
                user._info[2] = requested_service_life
                user._info[5] = service_quantity

    def move_users(self, current_iteration):
      
        for user_idx in range(self.n_users):
            if (current_iteration<self.k_steps_to_walk):
                self.users[user_idx]._x_coord = self.users_walk_steps[user_idx][current_iteration][0]/CELL_RESOLUTION_PER_COL
                self.users[user_idx]._y_coord = self.users_walk_steps[user_idx][current_iteration][1]/CELL_RESOLUTION_PER_ROW
                self.users[user_idx]._z_coord = self.users_walk_steps[user_idx][current_iteration][2]

    def compute_users_walk_steps(self):
        
        min_steps = 2
        max_steps = 5
        k_steps = np.random.random_integers(min_steps, max_steps)
        self.k_steps_to_walk = k_steps
        
        users_walks = scenario_objects.User.k_random_walk(self.users, k_steps)
        self.users_walk_steps = users_walks

        if (FIXED_CLUSTERS_NUM>0):
            self.clusterer, self.users_clusters = scenario_objects.User.compute_clusterer(self.users) 
        else:
            optimal_clusterer, users_clusters, optimal_clusters_num, current_best_silhoutte_score = scenario_objects.User.compute_clusterer(self.users, fixed_clusters=False)
            self.clusterer = optimal_clusterer
        
        self.cluster_centroids = scenario_objects.User.actual_users_clusters_centroids(self.clusterer)
        self.clusters_radiuses = scenario_objects.User.actual_clusters_radiuses(self.cluster_centroids, self.users_clusters, FIXED_CLUSTERS_NUM)
        
