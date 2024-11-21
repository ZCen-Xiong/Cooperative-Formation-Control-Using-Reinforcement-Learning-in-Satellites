import random
import numpy as np
import os
import pickle
# SAC config
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None) #没有充满时先用none创造出self.position
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity #超出记忆池容量后从第一个开始改写以维持容量不超标
 
    def sample(self, batch_size, seed_r = None):
        # if seed_r is None:
        #     random.seed(42)
        # else:
        #     random.seed(seed_r)
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        '''zip()函数用于将多个可迭代对象（例如列表、元组等）的对应元素打包成一个元组，返回一个迭代器。
        map()函数将一个函数应用于迭代器中的每个元素，并返回一个新的迭代器。
        这里将np.stack()函数应用于zip(*batch)的每个元素。np.stack()函数用于沿着新的轴堆叠数组序列，返回一个新的数组。
        map(np.stack, zip(*batch))将返回一个包含多个新数组的迭代器，其中每个新数组由批次中对应元素的堆叠组成。
        最后将上一步得到的迭代器中的新数组分别赋值给state、action、reward、next_state和done这五个变量。这意味着每个变量都是一个包含了批次中对应元素的堆叠数组。
        【迭代器】：允许按需逐个访问集合中的元素，而不是一次性获取整个集合；range()函数、zip()函数和字典的items()方法都返回迭代器，
        还可以使用关键字yield来定义生成器函数，生成器函数返回的对象也是迭代器'''
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
        # 双下划线（__）用于表示特殊方法或特殊属性。这些特殊方法和属性具有预定义的名称，它们在对象上具有特殊的行为。
        # __len__() 是一个特殊方法，用于返回对象的长度

    def save_buffer(self, env_name, suffix="", save_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')

        if save_path is None:
            save_path = "checkpoints/sac_buffer_{}_{}".format(env_name, suffix)
        print('Saving buffer to {}'.format(save_path))

        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, save_path):
        print('Loading buffer from {}'.format(save_path))

        with open(save_path, "rb") as f:
            self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity