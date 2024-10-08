import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt
import os
import math
import time
from Orbit_Dynamics.CW_Prop import Free, orbitgen
from control_dep import discrete_state ,MPC_dep, quadprog
'''一个无重力环境下的三维测试用环境'''

class Rel_trans:
    def __init__(self, sma, thrust, ref_p, T_f, discreT,Cd,R,Q11,Q22,S11,S22,pre_horizon):
        # 输入单位为m和s 半长轴 推力加速度 相对轨道振幅
         #状态空间设置为无限连续状态空间，虽然不知道相比设成离散空间有什么影响
        self.sma = sma # semi major axis
        self.mu = 398600.5e9
                # scaling
        self.c_time = np.sqrt(self.mu / (self.sma ** 3))
        # self.nT_real = np.sqrt(self.mu / (self.sma ** 3))   # real angular velocity
        self.c_dis = self.mu**(1/3)/self.sma
        # normalized
        self.nT = 1 
        self.T_final = T_f*self.c_time
        self.dT_norm = discreT*self.c_time
        self.thrust_norm = thrust*self.c_dis/(self.c_time**2)
        self.p_norm = ref_p*self.c_dis
        
        # prediction horizon is xxx steps
        self.horizon = pre_horizon
        # action 包括a和b两个方向，预测h步
        self.action_space=spaces.Box(low = -0.1 , high = 0.1, shape=(2*self.horizon,), dtype=np.float32) 
        # 实际观测状态是当前状态+目标状态 3个智能体x当前+目标状态 6个参数 +3(空间指向)+ 时间
        self.observation_space=spaces.Box(-np.inf,np.inf,shape=(3*2*6+3+1,),dtype=np.float32)
        # current state 三个星的状态
        self.agent_state = np.zeros((3*6,))
        # 当前的轨道序列
        self.agent_seq = np.zeros((3*6*self.horizon,))
        # 无智能体干预的轨道序列 
        # self.dumm_seq = np.zeros((3*6*self.horizon,)) # 0425 这玩意没有意义，预测用不上，规划动作用不上，输出奖励用不上
        # 目标轨道的序列
        self.target_seq = np.zeros((3*6*self.horizon,)) 

        self.Ad_norm, self.Bd_norm = discrete_state(self.dT_norm,self.nT)
        self.Cd_norm = Cd # Cd是读出矩阵，保持不变
        # Q矩阵（状态的费用）
        Q11 = Q11/ (self.c_dis**2)
        Q22 = Q22 / (self.c_dis**2 / (self.c_time**2))
        self.Q_norm = np.block([[Q11, np.zeros((3, 3))], [np.zeros((3, 3)), Q22]])
        # S 结束状态的费用
        S11 = np.eye(3) /  (self.c_dis**2)
        S22 = np.eye(3) / (self.c_dis**2 / (self.c_time**2))
        self.S_norm = np.block([[S11, np.zeros((3, 3))], [np.zeros((3, 3)), S22]])
        # R矩阵（控制的费用）
        self.R_norm = R / (self.c_dis**2 / (self.c_time**4))
        # 生成MPC各种矩阵
        # 状态和量测维度按道理应该分别取 state的维数 和 C的行数 
        # self.agent_state.shape[0], self.Cd_norm.shape[0]
        self.MPC = MPC_dep(6,6, self.horizon, 
                           self.Ad_norm, self.Bd_norm, self.Cd_norm, 
                           self.R_norm, self.Q_norm, self.S_norm)
        self.Q_bar = self.MPC.Qbar_func()
        self.R_bar = self.MPC.Rbar_func()
        # U对状态影响 AB联合
        self.F_norm = self.MPC.Fbar_func()
        #  A联合
        self.M_norm = self.MPC.Mbar_func()
        Hbar = self.MPC.H_func()
        self.Hbar_half = Hbar/2 + Hbar.T/2
        self.f = self.MPC.f_func()
        '''数据排列规则.由于实际输出的全是长一维向量.需要按照这个规则来确定智能体和预测的状态是第几个'''
        self.s_len = 6*self.horizon # 智能体数据间的间隔，意思就是每个智能体有多少步
        self.r_len = 6              # 时间相邻两个状态的数据间隔

    def reset(self, ini_fin_in = None,ini_phi = None, seed = None): #,prop_t
        super().__init__()
        '''固定随机数种子'''
        self.adj_seed = 0
        if seed is not None:
            np.random.seed(seed)
            self.adj_seed = seed
        else:
            np.random.seed(int(time.time()*1000%1000))
        self.agent_state_list=[]
        self.t_scn = 0 # 场景初始时间
        self.isInject_n=[False for _ in range(3)] # done无法用于区分是否到达目标，所以加入另一个标志位
        self.done_n=[False for _ in range(3)] # done是q函数计算所必要的
        if ini_fin_in is None:
            ''' initial orbit 以下为随机生成两组差距36度的轨道'''
            self.azi_0 = np.random.uniform(0, 2*np.pi)
            # 只朝z正已经覆盖了全部目标空间了，因为智能体姿态可以正负转，主要害怕的是-0到+0的奇异
            self.ele_0 = np.random.uniform(0.05*np.pi, 0.5*np.pi)
            # final orbit 轨道差为0.2pi和0.1pi
            self.azi_f = self.azi_0 + np.random.uniform(-0.1*np.pi, 0.1*np.pi)
            self.ele_f = self.ele_0 + np.random.uniform(-0.1*np.pi, 0.1*np.pi)
            '''固定实验 '''
            if self.ele_f > 0.5*np.pi or self.ele_f < 0.1*np.pi:
                self.ele_f = self.ele_0 - (self.ele_f - self.ele_0)
        else:
            self.azi_0 = ini_fin_in[0]
            self.ele_0 = ini_fin_in[1]
            self.azi_f = ini_fin_in[2]
            self.ele_f = ini_fin_in[3]

        # Check and adjust self.ele_f if it exceeds 0.5*pi

        # real amplitude =  20 * sma /(mu)**(1/3)
        self.orbit_i0 = orbitgen(self.nT , self.azi_0, self.ele_0, self.p_norm)
        self.orbit_f0 = orbitgen(self.nT , self.azi_f, self.ele_f, self.p_norm)
        # 生成初始轨道的随机相位
        if ini_phi is None:
            phase_d = np.random.uniform(0.0,2*np.pi)
        else:
            phase_d = ini_phi
        # 求出轨道状态
        Xi_0 = Free(phase_d,0,self.orbit_i0,self.nT)
        Xf_0 = Free(phase_d,0,self.orbit_f0,self.nT)
        theta_i = np.arctan2(Xi_0[1], Xi_0[0])
        theta_f = np.arctan2(Xf_0[1], Xf_0[0])
        theta_diff = theta_f - theta_i  # 顺时针下，f滞后于i的相位
        f_phase_adjust = 0
        if np.abs(theta_diff) > np.pi:  # 说明超出范围了,因为我在生成轨道的是都只会差一点角度
            real_phd = 2*np.pi - np.abs(theta_diff)
            if theta_f < 0:
                f_phase_adjust = 2*real_phd
        elif theta_diff > 0:
            f_phase_adjust = 2*theta_diff
        Xf_adj = Free(f_phase_adjust,0,Xf_0,self.nT)
        Bi_0 = Free(np.pi*2/3,0,Xi_0,self.nT)
        Ci_0 = Free(-np.pi*2/3,0,Xi_0,self.nT)
        Bi_f = Free(np.pi*2/3,0,Xf_adj,self.nT)
        Ci_f = Free(-np.pi*2/3,0,Xf_adj,self.nT)

        # 尤其需要注意，sat_state是 t时刻的[智能体1,智能体2，智能体3]
        # 而后面的 agent_seq 是 [智能体1（1-5时刻）,智能体2（1-5时刻），智能体3（1-5时刻）]
        # 这里的状态用于外推，不用于输入智能体
        self.agent_state = np.concatenate((Xi_0, Bi_0, Ci_0))
        self.dumm_state = np.concatenate((Xi_0, Bi_0, Ci_0))
        self.target_state = np.concatenate((Xf_adj, Bi_f, Ci_f))

        def an2vec(elevation_rad,azimuth_rad):
            # 计算方向矢量
            x = np.cos(elevation_rad) * np.cos(azimuth_rad)
            y = np.cos(elevation_rad) * np.sin(azimuth_rad)
            z = np.sin(elevation_rad)
            # 输出方向矢量
            direction_vector = np.array([x, y, z])
            return direction_vector
        
        '''智能体观测空间的输入.当前状态.当前目标.指向.时间历程'''
        self.heading = an2vec(self.ele_f,self.azi_f)
        self.travel = np.zeros(1)  
        
        self.permutations = [
            [0, 1, 2],  # 123排列
            [1, 2, 0],  # 231排列
            [2, 0, 1]   # 312排列
        ]
        
        # 除以1000，在GEO下，200km的归一化长度为300左右, 包括目标指向的矢量，我希望网络能学到其中的关联
        obs_agent_state = [Xi_0,Bi_0,Ci_0]
        obs_target_state = [Xf_adj,Bi_0,Ci_0]

        return self.make_obs_n(obs_agent_state,obs_target_state)
    '''obs顺序
         0   cnt   17 18  tgt   35  36 hed 38   39
    obs1:[-0- -1- -2-][-0- -1- -2-] 
    obs2:[-1- -2- -0-][-1- -2- -0-]
    obs3:[-2- -1- -0-][-2- -1- -0-]
    '''
    def make_obs_n(self,obs_agent_state, obs_target_state):
        obs_make = []
        for perm in self.permutations:
            # 根据排列顺序获取W中的元素，注意不要忘了放缩
            obs_agent_temp = np.array([obs_agent_state[i] for i in perm])/1e3
            # 根据排列顺序获取Y中的元素
            obs_target_temp = np.array([obs_target_state[i] for i in perm])/1e3
            # 将W和Y的排列组合合并为一个数组
            combined_array = np.concatenate((obs_agent_temp.ravel(), obs_target_temp.ravel(),self.heading,self.travel))
            # 将合并后的数组添加到列表中
            obs_make.append(combined_array)
    
        return obs_make

    def internal_step(self, Sat_action, isPlot):
        # action是一个向量系数 3*2*self.horizon 个 3个智能体 每个智能体2个维度
        # 前一半的系数3*horizon是alpha，后一半的系数3*horizon是beta        
        self.agent_state_list.append(self.agent_state) # N x 18的矩阵

        # 预测序列
        for sat_i in range(0,3):
            for ri in range(0,self.horizon):
                # 预测序列是按这样布置的，先把第一个智能体的1-5步放完，再把第二个智能体的1-5步放完，再放第三个智能体的1-5步
                #   sat1 t1-t5, sat2 t1-t5, sat3 t1-t5
                t_r = (ri+1)*self.dT_norm
                # 这种就是按照 先排完一个智能体的全部数据，再排另外一个智能体
                index_start = sat_i*self.s_len + ri*self.r_len  # 0 
                index_end = index_start + 6
                self.agent_seq[index_start:index_end] = Free(t_r,0,self.agent_state[sat_i*6:sat_i*6+6],self.nT)
                self.target_seq[index_start:index_end] = Free(t_r,0,self.target_state[sat_i*6:sat_i*6+6],self.nT)
        
        alpha_list =[]
        beta_list = []
        for agt_idx in range(3):
            alpha_i = np.repeat(Sat_action[agt_idx][0:self.horizon],6)
            beta_i = np.repeat(Sat_action[agt_idx][self.horizon:],6)  # 3 * horizon
            alpha_list.append(alpha_i)
            beta_list.append(beta_i)
        
        alpha = np.concatenate(alpha_list)
        beta = np.concatenate(beta_list)
        # 沿着法向指向的矢量    
        normal_track = self.agent_seq - self.target_seq  # 第一个智能体h*6个状态 第二个智能体 h*6个状态 第三个智能体 h*6个状态        
        # 沿着径向指向的矢量 由目标状态指向原点
        radius_track = - self.target_seq    # 3 * horizon * 6 第一个智能体h*6个状态 第二个智能体 h*6个状态 第三个智能体 h*6个状态
        
        # 未修正的路径航点  alpha偏置，beta偏置 + 当前预测点
        raw_point = np.dot(alpha,normal_track) + np.dot(beta,radius_track)*1e-2
        # 里程
        self.travel[0] = self.t_scn/self.T_final
        
        '''目标点的规划  self.travel[0]**(1/2.5)'''
        target_weight = self.travel[0]
        # 修正后的路径航点 3*(h*6)
        pred_point = raw_point * (1-target_weight) + self.target_seq
        # MPC序列 的控制上下界
        lb = -np.ones([3*self.horizon,1])*self.thrust_norm
        ub = np.ones([3*self.horizon,1])*self.thrust_norm

        '''现实状态'''
        next_agent_state = np.zeros([3*6,]) # 18个状态，就是最正统的智能体在T_SCN控制后的实际状态
        next_dumm_state = np.zeros([3*6,]) # 不使用强化学习的MPC规划
        next_target_state = np.zeros([3*6,])
        ''' 现实面积'''
        next_agent_pos = np.zeros([3,3])    # 仅用于计算三颗星的方向与目标天区
        next_dumm_pos = np.zeros([3,3])
        '''seq 仅用于计算是否入轨.这里解释下, 
          为什么只有dumm 和 agent有seq? 因为前面算的seq都是自由漂移的seq,
          这里储存受控的seq,但target并不需要这种seq, 它不受控'''
        next_obs_agt = []
        next_obs_tgt = []
        dV_rew = np.zeros(3)
        J_agt = []
        J_dum = []

        for sat_i in range(3):
            # 从target集合中，取出单个智能体的预测序列 
            # 取出第一个智能体的 6*horizon 目标状态
            s_target_seq = pred_point[self.s_len*sat_i:self.s_len*(sat_i+1)] 
            # 目标不偏置状态
            s_orgn_seq = self.target_seq[self.s_len*sat_i:self.s_len*(sat_i+1)]
            # 取出第一个智能体的 6*1当前状态
            s_sat_state = self.agent_state[sat_i*6:sat_i*6+6]
            s_dumm_state = self.dumm_state[sat_i*6:sat_i*6+6] # 全程MPC
            # s_dumm_state = self.agent_state[sat_i*6:sat_i*6+6]   # 阶段MPC ( 就是一坨屎 )
            s_target_state = self.target_state[sat_i*6:sat_i*6+6]
            # 一次项系数
            fo = np.hstack((s_sat_state, s_target_seq)).T @ self.f
            fo_dumm = np.hstack((s_dumm_state, s_orgn_seq)).T @ self.f
            # 计算出第一个智能体的 3*horizon 个机动控制
            Acons_norm = self.MPC.Acons_func() @ self.F_norm
            Bcons_norm = self.MPC.Dy_func(-np.ones((6,1))*1e2, np.ones((6,1))*1e2) - self.MPC.Acons_func()@ self.M_norm @ s_sat_state.reshape(6, 1)
            s_agent_act_seq = quadprog(self.Hbar_half, fo.T, Acons_norm, Bcons_norm,  None, None,lb, ub).flatten()  
            # 计算出没有智能体参与的 3*horizon 个机动控制
            s_dumm_act_seq = quadprog(self.Hbar_half, fo_dumm.T, Acons_norm, Bcons_norm,  None, None,lb, ub).flatten()  
            # s_agent_act_seq = s_agent_act_seq.flatten() # 3*horizon
            # current_action = s_agent_act_seq[0:3]
            
            next_agent_state[6*sat_i:6*(sat_i+1)] = self.Ad_norm @ s_sat_state + self.Bd_norm @ s_agent_act_seq[0:3]
            next_dumm_state[6*sat_i:6*(sat_i+1)] = self.Ad_norm @ s_dumm_state + self.Bd_norm @ s_dumm_act_seq[0:3]
            next_target_state[6*sat_i:6*(sat_i+1)] = self.Ad_norm @ s_target_state
            '''插一个观测空间的读取'''
            next_obs_agt.append(next_agent_state[6*sat_i:6*(sat_i+1)])
            next_obs_tgt.append(next_target_state[6*sat_i:6*(sat_i+1)])
            dV_rew[sat_i] = np.linalg.norm(s_agent_act_seq[0:3])
            # 仅用于计算三颗星的方向与目标天区
            next_agent_pos[sat_i,:] = next_agent_state[6*sat_i:6*sat_i+3]
            next_dumm_pos[sat_i,:] = next_dumm_state[6*sat_i:6*sat_i+3]
            # 仅用于计算是否入轨
            agt_seq_tmp = self.M_norm @ self.agent_state[6*sat_i:6*(sat_i+1)] + self.F_norm @ s_agent_act_seq
            dum_seq_tmp = self.M_norm @ self.dumm_state[6*sat_i:6*(sat_i+1)] + self.F_norm @ s_dumm_act_seq
            agt_err = agt_seq_tmp - s_orgn_seq
            dum_err = dum_seq_tmp - s_orgn_seq
            J_agt.append(np.sqrt(agt_err @ self.Q_bar @ agt_err.T))
            J_dum.append(np.sqrt(dum_err @ self.Q_bar @ dum_err.T))

        '''计算目标天区与三颗星的方向构成面积的奖励'''
        def calculate_area(pos):
            link1 = pos[1,:] - pos[0,:]
            link2 = pos[2,:] - pos[0,:]
            formation_product = np.cross(link1, link2)
            proj_area = abs(0.5 * np.dot(formation_product, self.heading))
            return proj_area
        
        heading_reward = calculate_area(next_agent_pos) - calculate_area(next_dumm_pos)
        self.agt_area = calculate_area(next_agent_pos)
        self.dum_area = calculate_area(next_dumm_pos)
        self.dumm_state = next_dumm_state
        self.agent_state = next_agent_state
        self.target_state = next_target_state
        reward = [0,0,0]
        if isPlot is None:

            """ reward here 随着步数递减 """
            '''纯面积奖励'''
            # reward =  heading_reward/10 
            '''慢于MPC则出现控制惩罚'''
            exc_t_puni = 0.0
            exc_t_rewa = 1e-2
            
            Conv_tol = 10
            for sat_i in range(3):
                if np.linalg.norm(J_dum[sat_i])< Conv_tol:
                    exc_t_puni = 1e-1
                if np.linalg.norm(J_dum[sat_i])<1e-10:
                    exc_t_rewa = 0.0

                # un_step = self.dT_norm/self.T_final
                # reward[sat_i] = exc_t_rewa * 10./(J_agt[sat_i]-J_dum[sat_i])* heading_reward - \
                #     exc_t_puni*self.travel[0]*(1e-2*dV_rew[sat_i] - 10./(J_agt[sat_i]-J_dum[sat_i]))
                reward[sat_i] = exc_t_rewa * (1 - self.travel[0]**(1/2)) * heading_reward - \
                    exc_t_puni*self.travel[0]*(1e-2*dV_rew[sat_i] - 10./(1 + J_agt[sat_i]-J_dum[sat_i]))
                #训练线性模数用的价值
                # reward[sat_i] = exc_t_puni*self.travel[0]*( - np.linalg.norm(Sat_action)/3 + 5./(J_agt[sat_i]/J_dum[sat_i]))

                if self.t_scn >= self.T_final or J_agt[sat_i] <= Conv_tol:
                    self.done_n[sat_i] = True
                    if J_agt[sat_i] > Conv_tol:
                        reward[sat_i] -= 1
                    elif J_agt[sat_i] <= Conv_tol and self.t_scn <= self.T_final:
                        if self.isInject_n[sat_i] is False:
                            reward[sat_i] += 100
                            self.isInject_n[sat_i]= True
        else: # 绘图仅输出面积奖励
            reward[sat_i] = heading_reward

        '''放在这里的原因是：例如这个函数输入是第一步，那么234到235行已经计算出第2步了，
            对于agent，并不要第二步作为输入, 动作是3-7步，而在267行已经+1了，所以时间外推不能放前面'''
        self.t_scn += self.dT_norm
        next_obs_n = self.make_obs_n(next_obs_agt,next_obs_tgt)

        return next_obs_n, reward, self.done_n, self.isInject_n

    def step(self,Sat_action): 
        next_obs_n, reward_n, done_n, isInject_n = self.internal_step(Sat_action, isPlot = None)
        return next_obs_n, reward_n, done_n, isInject_n
    
    def plotstep(self,Sat_action):
        next_obs_n, h_reward_n, done_n, isInject_n = self.internal_step(Sat_action, isPlot = 1)
        return next_obs_n, h_reward_n, done_n, isInject_n
    
    def plot(self, args, data_x, data_y, data_z=None):
        if data_z!=None and args['plot_type']=="2D-2line":
            fig = plt.figure()
            ax = fig.gca() #Axes对象是图形的绘图区域，可以在其上进行各种绘图操作。通过gca()函数可以获取当前图形的Axes对象，如果该对象不存在，则会创建一个新的。
            plt.plot(data_x,data_y,'b',linewidth=0.5)
            plt.plot(data_x,data_z,'g',linewidth=1)
            ax.set_xlabel('x', fontsize=15)
            ax.set_ylabel('y', fontsize=15)
            ax.set_xlim(np.min(data_x),np.max(data_x))
            ax.set_ylim(np.min([np.min(data_y),np.min(data_z)]),np.max([np.max(data_y),np.max(data_z)]))
            if not os.path.exists('logs'):
                os.makedirs('logs')
            plt.savefig(args['plot_title'])# 'logs/{}epoch-{}steps.png'.format(epoch,steps)
