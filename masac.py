# 多智能体修改， memory直接包括了三个状态，动作和奖励
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update
from model import GaussianPolicy, QNetwork, DeterministicPolicy
# SAC 主程序
class MASAC(object):
    def __init__(self, num_inputs, action_space, args, num_ff):
        # numinputs 是观测空间大小
        self.gamma = args['gamma']
        self.tau = args['tau']
        self.alpha = args['alpha']
        self.num = num_ff
        self.policy_type = args['policy']
        self.target_update_interval = args['target_update_interval']
        self.automatic_entropy_tuning = args['automatic_entropy_tuning']

        self.device = torch.device("cuda" if args['cuda'] else "cpu")
        # print(self.device)
        self.critic = QNetwork(num_inputs, num_ff * action_space.shape[0], args['hidden_sizes'], args['activation']).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args['lr'])

        self.critic_target = QNetwork(num_inputs, num_ff * action_space.shape[0], args['hidden_sizes'], args['activation']).to(self.device)
        hard_update(self.critic_target, self.critic) #初始化的时候直接硬更新

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True: #原论文直接认为目标熵就是动作空间维度乘积的负值，在这里就是Box的“体积”
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item() #torch.prod()是一个函数，用于计算张量中所有元素的乘积
                self.alpha = torch.zeros(1, requires_grad=True, device=self.device) #原论文没用log，但是这里用的，总之先改成无log状态试试
                #self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device) #初始化log_alpha
                self.alpha_optim = Adam([self.alpha], lr=args['lr'])

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args['hidden_sizes'], args['activation'], action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args['lr'])
        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args['hidden_sizes'], action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args['lr'])

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state) #如果evaluate为True，输出的动作是网络的mean经过squash的结果
        return action.detach().cpu().numpy()[0]

    def update_parameters(self,agt_idx, memory_n, batch_size, updates, oth_agt):
        # Sample a batch from memory
        state_batch_n, action_batch_n, reward_batch, next_state_batch_n, done_batch = memory_n.sample(batch_size=batch_size, seed_r = updates)
        # 由于memory里的_n变量 里面是没有区分, sb_[0] 是当前序号的（例如2），sb_2 递推（则是0）
        sb_ = [None for _ in range(self.num)]
        ab_ = [None for _ in range(self.num)]
        nsb_ = [None for _ in range(self.num)]
        '''这里序号很绕，解释下，我把三个智能体的观测空间都输入进来了，但是对于up_para函数里，第一个智能体的序号其实是agt_idx
            比如是2吧，那up_para里第二个智能体的编号其实就该是0了，因为是201的顺序
        '''
        for idx in range(self.num):
            # 由于sample函数里用了个奇怪的map，导致编号变成了第二个index,所以用 :,a,:来提取
            sb_[idx] = torch.FloatTensor(state_batch_n[:,(agt_idx + idx)%self.num,:]).to(self.device)
            ab_[idx] = action_batch_n[:,(agt_idx + idx)%self.num,:]
            nsb_[idx] = torch.FloatTensor(next_state_batch_n[:,(agt_idx + idx)%self.num,:]).to(self.device)
        # 这里的所有batch都是一整个数组
        ab_n = np.hstack((ab_[0],ab_[1],ab_[2]))
        # ab_1 = torch.FloatTensor(ab_1).to(self.device)
        ab_n = torch.FloatTensor(ab_n).to(self.device)        
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        done_batch = torch.FloatTensor(done_batch).to(self.device).unsqueeze(1)
        
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(nsb_[0]) #policy网络算出来的action   
            '''infer'''
            # _, _, nsa2 = self.policy.sample(nsb_[1]) #policy网络算出来的action   
            # _, _, nsa3 = self.policy.sample(nsb_[2]) #policy网络算出来的action   
            _, _, nsa2 = oth_agt[(agt_idx+1)%3].policy.sample(nsb_[1]) #policy网络算出来的action   
            _, _, nsa3 = oth_agt[(agt_idx+2)%3].policy.sample(nsb_[2]) #policy网络算出来的action   
            nsa_n =  torch.cat((next_state_action, nsa2, nsa3), dim=1)
            '''infer end'''

            qf1_next_target, qf2_next_target = self.critic_target(nsb_[0], nsa_n) #target算出来的q值
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) #选择较小的Q值
            target_q_value = reward_batch + (1 - done_batch) * self.gamma * (min_qf_next_target - self.alpha * next_state_log_pi) #原论文(2),(3)式
            # 上式为bellman backup,备份一个状态 或是状态动作对，是贝尔曼方程的右边，即reward+next value
        qf1, qf2 = self.critic(sb_[0], ab_n)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, target_q_value)  # MSEloss是对一个batch中所有样本的loss取差值平方后求平均
        qf2_loss = F.mse_loss(qf2, target_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward() #这里的qf_loss保留了梯度信息而非简单相加，因此(loss1+loss2)整体对两个网络做梯度反向传播时，loss2对q1网络的梯度为0
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(sb_[0])
        '''infer 被推测项 不提供熵项'''
        # _, _, pi_2 = self.policy.sample(sb_[1])
        # _, _, pi_3 = self.policy.sample(sb_[2])
        # 使用平均化输出作为对其他的估计，此时使用的网络是真的其他的网络
        _, _, pi_2 = oth_agt[(agt_idx+1)%3].policy.sample(sb_[1])
        _, _, pi_3 = oth_agt[(agt_idx+2)%3].policy.sample(sb_[2])
        # pi_2 = torch.FloatTensor(ab_[1]).to(self.device) 
        # pi_3 = torch.FloatTensor(ab_[2]).to(self.device) 
        pi_n = torch.cat((pi, pi_2, pi_3),dim=1)
        '''infer end'''

        qf1_pi, qf2_pi = self.critic(sb_[0], pi_n)

        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))] 原论文式(7)

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.alpha * (log_pi + self.target_entropy).detach()).mean() #原论文里的J函数就是loss，不需要再在代码里给出∇形式

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau) #对目标网络软更新

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_checkpoint(self, ckpt_path=None):
        '''if not os.path.exists('GoodModel/'):
            os.makedirs('GoodModel/')'''
        if ckpt_path is None:
            ckpt_path = "sac_scene1_attack.pt".format()
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict()}, ckpt_path)

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])
            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()

