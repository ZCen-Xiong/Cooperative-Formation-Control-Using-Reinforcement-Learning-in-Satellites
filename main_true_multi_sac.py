'''
Soft Actor-Critic version 2
using target Q instead of V net: 2 Q net, 2 target Q net, 1 policy net
add alpha loss compared with version 1
paper: https://arxiv.org/pdf/1812.05905.pdf
要点：1.输出随机策略，泛化性强（和之前PPO的思路一样，其实是正确的）
2.基于最大熵，综合评价奖励值与动作多样性（-log p(a\s)表示熵函数，单个动作概率越大，熵函数得分越低；为了避免智能体薅熵函数分数（熵均贫化），
可以动态改变熵函数系数（温度系数），先大后小，实现初期注重探索后期注重贪婪
3.【重点】随机策略同样是网络生成方差+均值，和“以为PPO能做的”一样，但是必须用重参数化，即不直接从策略分布中取样，而是从标准正态分布
N(0,1)中取样i，与网络生成的方差+均值mu,sigma得到实际动作a=mu+sigma*i，这样保留了参数本身，才能利用链式法则求出loss相对参数本身的梯度；
如果直接取样a，a与参数mu和sigma没有关系，根本没法求相对mu和sigma的梯度；之前隐隐觉得之前的PPO算法中间隔了个正态分布所以求梯度这一步存在问题其实是对的...
4.目前SAC实现的算法（openAI和作者本人的）都用了正态分布替代多模Q函数，如果想用多模Q函数需要用网络实现SVGD取样方法拟合多模Q函数（也是发明人在原论文中用的方法(Soft Q-Learning不是SAC））
'''

import datetime 
import numpy as np
import itertools
import torch.nn as nn
from masac import MASAC
#from torch.utils.tensorboard import SummaryWriter
from replay_mem_multi import ReplayMemory
import multi_env 
import time
import matplotlib.pyplot as plt
import numpy as np
from Orbit_Dynamics.CW_Prop import Free

# 字典形式存储全部参数
args={'policy':"Gaussian", # Policy Type: Gaussian | Deterministic (default: Gaussian)
        'eval':True, # Evaluates a policy a policy every 10 episode (default: True)
        'gamma':0.99, # discount factor for reward (default: 0.99)
        'tau':0.1, # target smoothing coefficient(τ) (default: 0.005) 参数tau定义了目标网络软更新时的平滑系数，
                     # 它控制了新目标值在更新目标网络时与旧目标值的融合程度。
                     # 较小的tau值会导致目标网络变化较慢，从而增加了训练的稳定性，但也可能降低学习速度。
        'lr':0.0003, # learning rate (default: 0.0003)
        'alpha':0.2, # Temperature parameter α determines the relative importance of the entropy\term against the reward (default: 0.2)
        'automatic_entropy_tuning':False, # Automaically adjust α (default: False)
        'batch_size':512, # batch size (default: 256)
        'num_steps':1000, # maximum number of steps (default: 1000000)
        'hidden_sizes':[1024,512,256], # 隐藏层大小，带有激活函数的隐藏层层数等于这一列表大小
        'updates_per_step':1, # model updates per simulator step (default: 1) 每步对参数更新的次数
        'start_steps':1000, # Steps sampling random actions (default: 10000) 在开始训练之前完全随机地进行动作以收集数据
        'target_update_interval':10, # Value target update per no. of updates per step (default: 1) 目标网络更新的间隔
        'replay_size':10000000, # size of replay buffer (default: 10000000)
        'cuda':True, # run on CUDA (default: False)
        'LOAD PARA': False, #是否读取参数
        'task':'Train',# 测试或训练或画图，Train,Test,Plot
        'logs':True, 
        'activation':nn.ReLU, #激活函数类型
        'plot_type':'2D-2line', #'3D-1line'为三维图，一条曲线；'2D-2line'为二维图，两条曲线
        'plot_title':'reward-steps.png',
        # 'seed':None, #网络初始化的时候用的随机数种子  
        'obgen_seed': 114514, # 轨道初始化种子
        'max_epoch':50000} #是否留存训练参数供tensorboard分析 

#Tensorboard
'''创建一个SummaryWriter对象，用于将训练过程中的日志信息写入到TensorBoard中进行可视化。
   SummaryWriter()这是创建SummaryWriter对象的语句。SummaryWriter是TensorBoard的一个API，用于将日志信息写入到TensorBoard中。
   format括号里内容是一个字符串格式化的表达式，用于生成一个唯一的日志目录路径。{}是占位符，format()方法会将占位符替换为对应的值
   datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")：这是获取当前日期和时间，并将其格式化为字符串。
   strftime()方法用于将日期时间对象转换为指定格式的字符串。在这里，日期时间格式为"%Y-%m-%d_%H-%M-%S"，表示年-月-日_小时-分钟-秒。
   "autotune" if args.automatic_entropy_tuning else ""：这是一个条件表达式，用于根据args.automatic_entropy_tuning的值来决定是否插入"autotune"到日志目录路径中。
   'runs/{}_SAC_{}_{}_{}'是一个字符串模板，其中包含了四个占位符 {}。当使用 format() 方法时，传入的参数会按顺序替换占位符，生成一个新的字符串。'''
#writer = SummaryWriter('runs/{}_SAC_{}_{}.txt'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),args['policy'], "autotune" if args['automatic_entropy_tuning'] else ""))
'''
显示图像：用cmd（不是vscode的终端） cd到具体存放日志的文件夹（runs），然后
    tensorboard --logdir=./
或者直接在import的地方点那个启动会话
如果还是不行的话用
    netstat -ano | findstr "6006" 
在cmd里查一下6006端口有没有占用，用taskkill全杀了之后再tensorboard一下
    taskkill /F /PID 26120
'''                    
# Environment
# env = NormalizedActions(gym.make(args.env_name))
if args['logs']==True:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter('./runs/')

sma = 7171e3
thrust = 5e-3
ref_p = 2e3
T_f = 2000
discreT = 50
Cd = np.eye(6)
R = np.eye(3)*1e6
Q11 = np.eye(3)*1e1
Q22 = np.eye(3)*2
S11 = np.eye(3)
S22 = np.eye(3)
num_ff = 3

env_f = multi_env.Rel_trans(sma, thrust, ref_p, T_f, discreT,Cd,R,Q11,Q22,S11,S22,pre_horizon = 5)

# 智能体和缓存区
agent_sac = [None for _ in range(num_ff)]
memory_n = [None for _ in range(num_ff)]
for agt_idx in range(num_ff):
    agent_sac[agt_idx] = MASAC(env_f.observation_space.shape[0], env_f.action_space, args, num_ff) #discrete不能用shape，要用n提取维度数量
    memory_n[agt_idx] = ReplayMemory(args['replay_size'])

if args['task']=='Train':
    # Training Loop
    updates = 0
    best_avg_reward = [0.0 for _ in range(3)]
    total_numsteps = 0
    best_single_done_num= [0 for _ in range(3)]
    if args['LOAD PARA']==True:
        for agt_idx in range(num_ff):
            agent_sac[agt_idx].load_checkpoint('model/sofarsogood_agt_{}.pt'.format(agt_idx))
            best_avg_reward[agt_idx] =  0.0
    
    # env_seed = np.random
    # env_seed.seed(114514)
    for i_episode in itertools.count(1): #itertools.count(1)用于创建一个无限迭代器。它会生成一个连续的整数序列，从1开始，每次递增1。

        success=False
        step_in_ep = 0
        episode_reward = 0.0
        agent_reward = [0.0 for _ in range(3)] # individual agent reward
        obs_n = env_f.reset(seed = i_episode + 100)
        while True:
            action_n = [None for _ in range(num_ff)]
            for agt_idx in range(num_ff):
                action_n[agt_idx] = agent_sac[agt_idx].select_action(obs_n[agt_idx],evaluate = False)  # 开始输出actor网络动作

            new_obs_n, rew_n, done_n, isInject_n= env_f.step(action_n) # Step

            for agt_idx in range(num_ff):
                # 使用sac_multi.py
                # memory_n[agt_idx].push(obs_n[agt_idx], action_n[agt_idx], rew_n[agt_idx], new_obs_n[agt_idx], done_n[agt_idx]) # Append transition to memory_n
                # 使用masac.py
                memory_n[agt_idx].push(obs_n, action_n, rew_n[agt_idx], new_obs_n, done_n[agt_idx]) 
            
            episode_reward += np.sum(rew_n)
            for i, rew in enumerate(rew_n): 
                agent_reward[i] += rew

            if len(memory_n[0]) > args['batch_size']:
                # Number of updates per step in environment 每次交互之后可以进行多次训练...
                for i in range(args['updates_per_step']):

                    for agt_idx in range(num_ff):
                        # 用sac_multi.py 是把三个memory一起塞进去采样，但是出现严重的采样问题。。。。
                        # critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent_sac[agt_idx].update_parameters(agt_idx, memory_n, args['batch_size'], updates)
                        # 使用masac.py 只塞一个memory, 但是每一个memory都包含了完整信息（见136行），而且数据顺序需要根据agt_idx读取来确定
                        critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent_sac[agt_idx].update_parameters(agt_idx, memory_n[agt_idx], args['batch_size'], updates, agent_sac)
                        # if args['logs']==True:
                        writer.add_scalar(f'agt{agt_idx}_loss/critic_1', critic_1_loss, updates)
                        writer.add_scalar(f'agt{agt_idx}_loss/critic_2', critic_2_loss, updates)
                        writer.add_scalar(f'agt{agt_idx}_loss/policy', policy_loss, updates)
                        writer.add_scalar(f'agt{agt_idx}_loss/entropy_loss', ent_loss, updates)
                        writer.add_scalar(f'agt{agt_idx}_entropy_temprature/alpha', alpha, updates)
                    updates += 1

            step_in_ep += 1
            obs_n = new_obs_n
            terminal = (step_in_ep >= T_f/discreT)
            if all(isInject_n) or terminal:
                if all(isInject_n):
                    success=True
                break

        if args['logs']==True:
            for idx in range(3):
                writer.add_scalar(f'agt_rwd{idx}', agent_reward[idx], i_episode)
            writer.add_scalar('reward_total/train', episode_reward, i_episode)            

        multireward = ', '.join(f"{round(rwd, 2)}" for rwd in agent_reward)
        print(f"Episode: {i_episode}, episode steps: {step_in_ep}, reward: [{multireward}], success: {success}")
            #   ,\
            #    azi:{round(env_f.azi_0, 2)}{round(env_f.azi_f, 2)},ele:{round(env_f.ele_0, 2)}{round(env_f.ele_f, 2)}")

        if i_episode % 100 == 0 and args['eval'] is True: #评价上一个训练过程
            avg_reward = [0. for _ in range(num_ff)]
            episodes_eval = 10
            done_num = 0
            single_done_num = [0 for _ in range(num_ff)]

            if args['LOAD PARA']==True:
                episodes_eval = 10

            for eval_ep  in range(episodes_eval):
                success=False
                step_in_ep = 0
                episode_reward = [0.0 for _ in range(3)]
                agent_reward = [0.0 for _ in range(3)] # individual agent reward
                obs_n = env_f.reset(seed = i_episode + eval_ep)
                while True:
                    action_n = [None for _ in range(num_ff)]
                    for agt_idx in range(num_ff):
                        action_n[agt_idx] = agent_sac[agt_idx].select_action(obs_n[agt_idx], evaluate = True)  # 开始输出actor网络动作

                    new_obs_n, rew_n, done_n, isInject_n= env_f.step(action_n) # Step

                    # episode_reward += np.sum(rew_n)
                    for i, rew in enumerate(rew_n): 
                        agent_reward[i] += rew

                    step_in_ep += 1
                    obs_n = new_obs_n
                    terminal = (step_in_ep >= T_f/discreT)
                    
                    if all(isInject_n) or terminal:
                        multireward = ', '.join(f"{round(rwd, 2)}" for rwd in agent_reward)
                        print(f'step:{step_in_ep}, reward:[{multireward}]')
                            #               ,\
                            #   azi:{round(env_f.azi_0, 2)}{round(env_f.azi_f, 2)},ele:{round(env_f.ele_0, 2)}{round(env_f.ele_f, 2)}')
                        for i, rew in enumerate(rew_n): 
                            single_done_num[i] += isInject_n[i]
                        if all(isInject_n):
                            success=True
                            done_num += 1
                        break

                for i, agt_rew in enumerate(agent_reward):
                    avg_reward[i] += agt_rew/episodes_eval
            
            # avg_reward /= episodes_eval
            #writer.add_scalar('avg_reward/test', avg_reward, i_episode)
            print("----------------------------------------")
            multireward = ', '.join(f"{round(rwd, 2)}" for rwd in avg_reward)
            print(f"Test Episodes: {episodes_eval}, Avg. Reward: [{multireward}],完成数：{done_num}")
            print("----------------------------------------")
            for i in range(num_ff):
                if avg_reward[i] > best_avg_reward[i] or best_single_done_num[i] < single_done_num[i]:
                    best_avg_reward[i] = avg_reward[i]
                    best_single_done_num[i] = single_done_num[i] 
                    agent_sac[i].save_checkpoint('model/sofarsogood_agt_{}.pt'.format(i))
                if single_done_num[i] >= episodes_eval:
                    agent_sac[i].save_checkpoint('model/tri_converged_{}.pt'.format(i))

            if done_num >= episodes_eval:
                from datetime import datetime
                print("完成训练,时间{}".format(datetime.now()))
                break

        if i_episode==args['max_epoch']:
            print("训练失败，{}次仍未完成训练".format(args['max_epoch']))
            if args['logs']==True:
                writer.close()
            break

if args['task']=='Test':
    # agent_sac.load_checkpoint("tri_agent.pt")
    for agt_idx in range(num_ff):
        agent_sac[agt_idx].load_checkpoint('model/tri_converged_{}.pt'.format(i))
        # agent_sac[agt_idx].load_checkpoint('model/sofarsogood_agt_{}.pt'.format(i))
    avg_reward = [0. for _ in range(num_ff)]
    episodes_eval = 10
    done_num = 0
    single_done_num = [0 for _ in range(num_ff)]

    if args['LOAD PARA']==True:
        episodes_eval = 10

    for eval_ep  in range(episodes_eval):
        success=False
        step_in_ep = 0
        episode_reward = [0.0 for _ in range(3)]
        agent_reward = [0.0 for _ in range(3)] # individual agent reward
        obs_n = env_f.reset(seed = eval_ep)
        while True:
            action_n = [None for _ in range(num_ff)]
            for agt_idx in range(num_ff):
                action_n[agt_idx] = agent_sac[agt_idx].select_action(obs_n[agt_idx], evaluate = True)  # 开始输出actor网络动作

            new_obs_n, rew_n, done_n, isInject_n= env_f.step(action_n) # Step

            # episode_reward += np.sum(rew_n)
            for i, rew in enumerate(rew_n): 
                agent_reward[i] += rew
                single_done_num[i] += isInject_n[i]

            step_in_ep += 1
            obs_n = new_obs_n
            terminal = (step_in_ep >= T_f/discreT )
            
            if all(isInject_n) or terminal:
                multireward = ', '.join(f"{round(rwd, 2)}" for rwd in agent_reward)
                print(f'step:{step_in_ep}, reward:[{multireward}]')
                if all(isInject_n):
                    success=True
                    done_num += 1
            break

        for i, agt_rew in enumerate(agent_reward): 
                avg_reward[i] += agt_rew/episodes_eval
    
    # avg_reward /= episodes_eval
    #writer.add_scalar('avg_reward/test', avg_reward, i_episode)
    print("----------------------------------------")
    multireward = ', '.join(f"{round(rwd, 2)}" for rwd in avg_reward)
    print(f"Test Episodes: {episodes_eval}, Avg. Reward: [{multireward}],完成数：{done_num}")
    print("----------------------------------------")
        
if args['task']=='Plot':
    for agt_idx in range(num_ff):
        agent_sac[agt_idx].load_checkpoint('model/tri_converged_{}.pt'.format(i))
    # Sat1_array,Sat2_array,Sat3_array = np.zeros((T_f/discreT,3))
    Satpos_array = np.zeros((3,np.ceil(T_f/discreT).astype(int),3))
    '''初末轨道'''
    in_fin_orbi = np.array([0.2,0.15,0.25,0.35])*np.pi
    '''相位'''
    ini_phi = 0.1*np.pi
    obs_n = env_f.reset(in_fin_orbi, ini_phi)
    agent_reward = [0.0 for _ in range(3)] # individual agent reward
    state_list = [obs_n[1]]
    step_in_ep = 0
    single_done_num = [0 for _ in range(num_ff)]
    while True:
        action_n = [None for _ in range(num_ff)]
        for agt_idx in range(num_ff):
            action_n[agt_idx] = agent_sac[agt_idx].select_action(obs_n[agt_idx], evaluate = True)  # 开始输出actor网络动作
        new_obs_n, h_rew_n, done_n, isInject_n= env_f.plotstep(action_n) # Step

        for i, rew in enumerate(h_rew_n): 
            agent_reward[i] += rew
            single_done_num[i] += isInject_n[i]

        step_in_ep += 1
        obs_n = new_obs_n
        state_list.append(obs_n[1])
        terminal = (step_in_ep >= T_f/discreT )
        
        if all(isInject_n) or terminal:
            multireward = ', '.join(f"{round(rwd, 2)}" for rwd in agent_reward)
            print(f'step:{step_in_ep}, reward:[{multireward}]')
            if all(isInject_n):
                success=True
        break

    print(agent_reward[i])
    tri_state = np.array(state_list)
    for i in range(tri_state.shape[0]):
        for sat_j in range(num_ff):
            Satpos_array[sat_j,i,:] = tri_state[i,sat_j*6:sat_j*6+3]
        # Sat2_array.append(plot_data[i][6:9]/1000)
        # Sat3_array.append(plot_data[i][12:15]/1000)
    '''plot the ini and final orbit'''
    ini_orb_state = env_f.orbit_i0
    fin_orb_state = env_f.orbit_f0
    t_period = np.arange(0, 2*np.pi, 0.05)
    Xi_orbit = np.zeros((6, len(t_period)))
    Xf_orbit = np.zeros((6, len(t_period)))
    for j in range(0, len(t_period)):
        Xi_orbit[:, j] = Free(t_period[j], 0, ini_orb_state, 1)
        Xf_orbit[:, j] = Free(t_period[j], 0, fin_orb_state, 1)
    fig = plt.figure()
    ax = fig.gca(projection='3d') 
    ax.plot(Xi_orbit[0, :], Xi_orbit[1, :], Xi_orbit[2, :], 'k')
    ax.plot(Xf_orbit[0, :], Xf_orbit[1, :], Xf_orbit[2, :], 'r')
    for sat_j in range(3):    
        ax.plot(Satpos_array[sat_j,0:step_in_ep,0],Satpos_array[sat_j,0:step_in_ep,1],Satpos_array[sat_j,0:step_in_ep,2],'b',linewidth=1) #画三维图
    ax.plot(0,0,0,'r*') #画一个位于原点的星形

    plt.show()
    # (args,Sat1_array[:][0],Sat1_list[:][1],Sat1_list[:][2])
    # (args,Sat2_list[:][0],Sat2_list[:][1],Sat2_list[:][2])
    # (args,Sat3_list[:][0],Sat3_list[:][1],Sat3_list[:][2])
    #writer.add_scalar('avg_reward/test', avg_reward, i_episode)
    # print("----------------------------------------")
    # print("完成：{}".format(done))
    # print("----------------------------------------")