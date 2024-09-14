import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
# SAC config
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        # 还是会执行n-1次，但循环最后一次（j=n-2）时激活函数是恒等映射
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)
    # 生成网络且允许灵活修改，但全都是全连接层，其中size可以是一串序列，每个元素都描述大小；同时j和j+1在循环中自动确保相乘时行数列数相等
    # nn.Identity 意味着网络的输出层将应用恒等映射作为激活函数，即输出值与输入值完全一致，没有经过任何变换
    # 灵活用星号解包
    # nn.Linear(a, b) 【不是一个单纯的全连接层】是 PyTorch 中的一个线性层（linear layer）的构造函数。它创建了一个将输入特征映射到输出特征的线性变换。
    # nn.Linear(a, b) 接受表示输入特征的维度a和输出特征的维度b，线性层的作用是通过学习一组权重和偏置，将输入特征进行线性变换，得到输出特征。
    # output = input * weight^T + bias
    # 其中，input 是输入特征，weight 是形状为 (b, a) 的权重矩阵，bias 是形状为 (b,) 的偏置项。^T 表示权重矩阵的转置。

class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_sizes,activation):
        super(ValueNetwork, self).__init__() #调用了 ValueNetwork 类的父类（nn.Module 类）的 __init__() 方法

        self.network=mlp([num_inputs] + list(hidden_sizes)+[1], activation)
        self.apply(weights_init_) #self.apply() 方法是 nn.Module 类中的一个方法，它会递归地遍历模型的所有子模块，并对每个子模块应用一个函数。
                                  #self.apply(weights_init_) 的作用是将自定义的参数初始化函数 weights_init_ 应用到 ValueNetwork 类的所有子模块上，从而初始化网络的参数。

    def forward(self, state):
        x = self.network(state)
        return x

class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_sizes,activation):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.Q_network_1=mlp([num_inputs+num_actions] + list(hidden_sizes)+[1], activation)

        # Q2 architecture
        self.Q_network_2=mlp([num_inputs+num_actions] + list(hidden_sizes)+[1], activation)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        x1 = self.Q_network_1(xu)
        x2 = self.Q_network_2(xu)
        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_sizes,activation, action_space=None,RE_PARAMETERIZATION=True):
        super(GaussianPolicy, self).__init__()
        self.pi_network=mlp([num_inputs]+list(hidden_sizes),activation,activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], num_actions)
        # 生成mu的层
        self.log_std_layer = nn.Linear(hidden_sizes[-1], num_actions)
        self.re_parameterization=RE_PARAMETERIZATION
        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x=self.pi_network(state)
        mean = self.mu_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        normal = Normal(mean, std)
        x_t = normal.rsample()
        # 【以下方案是代码作者自己的方案，先得到tanh动作再对这一动作求log】
        y_t = torch.tanh(x_t) # 没有做重参数化
        # print("Shape of self.action_scale:", self.action_scale.shape)
        # print("Shape of self.action_bias:", self.action_bias.shape)
        # print("Shape of y_t:", y_t.shape)
        action = y_t * self.action_scale + self.action_bias #不是重参数化，只是单纯把值调整到动作空间范围内
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log((1 - y_t.pow(2)) + epsilon) #试着去掉action_scale，根据图像不合理（如果y_t正好在0，在给予action_scale的限制后会变成0.7，导致还是会被减少log）
        # 原论文(21)式
        #log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon) #原论文中公式，但是多了个action_scale
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean
        # 【下面一段大注释是OpenAI的方案，先对x_t求完logprob再输出tanh之后的动作】
        # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
        # NOTE: The correction formula is a little bit magic. To get an understanding
        # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
        # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
        # Try deriving it yourself as a (very difficult) exercise. :)
        # 证明中用到了概率密度变换公式：y=g(x)，x,y均为随机变量，则概率密度函数有fy=fx·(dx/dy)（利用分布函数求导得概率密度很容易证明）
        # log_prob = normal.log_prob(x_t).sum(axis=-1)
        # # 从论文结果到代码实现见根目录附图
        # '''对于多维输出的情况，可以使用多维输出的 log 概率直接求和作为网络的 log 概率的原因是，假设各个维度之间是独立的。
        # 在连续动作空间中，策略网络通常输出一个多维向量，表示动作的各个维度。对于每个维度，可以假设输出是独立的，即每个维度的输出不受其他维度的影响。
        # 根据概率论中的乘法规则，当多个事件是独立的时，它们的联合概率可以通过各个事件的概率的乘积计算得到。
        # 因此，对于每个维度的动作，可以将其对应的 log 概率作为独立事件的概率，并将它们求和得到整体动作的 log 概率。
        # log(p1*p2*p3*p4)=logp1+logp2+logp3+logp4'''
        # log_prob -= (2*(np.log(2) - x_t - F.softplus(-2*x_t))).sum(axis=1)
        # # 用tanh限制正态分布范围，压到(-1,1)内
        # x_t = torch.tanh(x_t)
        # # 把动作扩到实际所需范围
        # action = self.action_scale * x_t + self.action_bias
        # mean = torch.tanh(mean) * self.action_scale + self.action_bias
        # return action, log_prob, mean
        

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)

class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)
