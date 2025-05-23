import torch
import torch.nn as nn
import numpy as np
from util_1 import plot_episode_rewards

# uncomment the following two lines if you are using wayland(Ubuntu) as the backend. Comment if not using wayland
import os
os.environ["QT_QPA_PLATFORM"] = "wayland"


class Net(nn.Module):
    def __init__(self, num_hidden=0, hidden_dim=50, state_dim=4, n_action=2):
        super(Net, self).__init__()
        ############################
        # Your Code #
        # Define the network.
        # The network include one input layer, num_hidden hidden layers, and one output layer.
        # Only use nn.Linear to define the layers. No other layers(Flatten, BatchNorm, etc.) are needed.
        
        self.input_layer = nn.Linear(state_dim, hidden_dim) # input layer

        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden)]) # hidden layers

        self.output_layer = nn.Linear(hidden_dim, n_action) # output layer
        ############################


    def forward(self, x):
        action_value = None
        ############################
        # Your Code #
        # Define the forward pass.
        # The forward pass includes the input layer, hidden layers, and output layer.
        # Between each layer, you need to apply the activation function using F.relu(x)
        # You do not need to add activation function after the output layer.

        x = nn.functional.relu(self.input_layer(x)) # input Layer + Activation Function
        
        for layer in self.hidden_layers: # for each input layer
            x = nn.functional.relu(layer(x)) # apply the activation function

        action_value = self.output_layer(x)  # final output layer with no activation
        ############################
        return action_value


class DQN(object):
    def __init__(self, target_replace_iter=10, batch_size=60, memory_capacity=500, num_hidden=1, hidden_dim=50, lr=0.01,
                 gamma=0.9, state_dim=4, n_action=2, soft_update_tau=0.01):
        self.batch_size = batch_size
        self.memory_capacity = memory_capacity
        self.target_replace_iter = target_replace_iter
        self.gamma = gamma
        # record the number of learning steps
        self.learn_step_counter = 0
        # record the size of memory
        self.memory_counter = 0
        # define the memory buffer to store the transitions
        self.memory = [None] * memory_capacity
        self.soft_update_tau = soft_update_tau
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        ############################
        # Your Code #
        # Define the q network and target network.
        # Move the net to the device
        # Optimizer: use Adam, set the learning rate as lr
        # Define loss function as MSELoss

        # Define the q network and target network. 
        self.q_net = Net(num_hidden, hidden_dim, state_dim, n_action).to(self.device) # Move the net to the device.
        self.target_net = Net(num_hidden, hidden_dim, state_dim, n_action).to(self.device) # Move the net to the device.
        
        # Optimizer: use Adam, set the learning rate as lr
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

        # Define loss function as MSELoss
        self.loss_func = nn.MSELoss()

        ############################

    def choose_action(self, x, epsilon, n_action):
        """
        choose action using epsilon-greedy method
        :param x: current state
        :param epsilon:
        :param n_action: number of actions
        :return:
        """
        # as the input, x is a numpy array, we need to convert it to a tensor
        x = torch.unsqueeze(torch.FloatTensor(x), 0).to(self.device)
        if np.random.uniform() < epsilon:
            action = np.random.randint(0, n_action)
        else:
            action = None
            ############################
            # Your Code #
            # If random value is greater than epsilon, take the action with highest Q value
            # First using the eval_net to get the action values
            # Then, using the max() method to get the action with the highest Q value
            # To take the action, you need to move the action value back to the cpu using .cpu()
            
            # Use the Q network to compute Q-values and choose the max
            action_values = self.q_net(x) # get action values
            action = torch.max(action_values, dim=1)[1].cpu().item() # choose max
            ############################

        return action

    def store_transition(self, s, a, r, s_):
        """
        store transition
        :param s: state
        :param a: action
        :param r: reward
        :param s_: next state
        :return:
        """
        index = None
        ############################
        # Your Code #
        # store the transition to the memory like a ring buffer.
        # If the memory is full, you need to replace the oldest transition with the new transition in the memory.
        # The code of transition replacing has been completed in the following 2 lines. You only need
        # to write one line to calculate the index of the oldest transition using the memory_counter and memory_capacity
        
        # Calculate the index in ring buffer, wrap around when full
        index = self.memory_counter % self.memory_capacity # ring‐buffer index

        ############################
        transition = [s, a, r, s_]
        self.memory[index] = transition

        self.memory_counter += 1

    def update_target_net(self, target_update_method='soft'):
        if target_update_method == 'hard':
            ############################
            # Your Code #
            # update the target network by copying the parameters from the eval network every target_replace_iter steps

            # Update the target network every target_replace_iter steps
            if self.learn_step_counter % self.target_replace_iter == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())
            ############################
            # you can comment the following line if you do not want to print the update information
            #print('hard update')
        elif target_update_method == 'soft':
            ############################
            # Your Code #
            # update the target network by using the soft update method. The formula is:
            # target_net = target_net * (1 - tau) + eval_net * tau
            # tau is the soft_update_tau
            # Tou need to use the state_dict() method to get the parameters of the network
            # The parameters of the network are stored in a dictionary. So you need to update the parameters for each key in the dictionary
            # You also need to use the load_state_dict() method to load the parameters to the target network

            # Soft update: target = target * (1 - tau) + q_net * tau
            q_params = self.q_net.state_dict() # get current parameters of q_net
            target_params = self.target_net.state_dict() # get current parameters of target_net

            for key in q_params: # for each parameter, apply the soft update function
                # Combine target network's current q-values with evaluation network q-values
                target_params[key] = ((1 - self.soft_update_tau) * target_params[key]) + (self.soft_update_tau * q_params[key])

            self.target_net.load_state_dict(target_params) # apply the changes
            ############################
            # you can comment the following line if you do not want to print the update information
            #print('soft update')
        else:
            raise ValueError('unknow target update method')

    def learn(self, target_update_method):
        """

        :param target_update_method:
        :return:
        """
        self.update_target_net(target_update_method)
        self.learn_step_counter += 1
        q_eval = q_target = None

        # sample batch size memory from the memory. We use np.random.choice to get a list of sampling indexes.
        sample_idx = np.random.choice(self.memory_capacity, self.batch_size)

        b_s = torch.FloatTensor([self.memory[idx][0] for idx in sample_idx]).to(self.device)
        b_a = torch.LongTensor([[self.memory[idx][1]] for idx in sample_idx]).to(self.device)
        b_r = torch.FloatTensor([[self.memory[idx][2]] for idx in sample_idx]).to(self.device)
        b_s_ = torch.FloatTensor([self.memory[idx][3] for idx in sample_idx]).to(self.device)

        ############################
        # Your Code #
        # 3 lines to compute the q value and target value.
        # line 1: get the values of the actions saved in b_a, based on the q network.
        #         You can use the .gather() method to get the values of the b_a together with the b_s.
        #         (Check https://pytorch.org/docs/stable/generated/torch.gather.html)
        # line 2: calculate the action values of the next state b_s_ through the target network and save them to q_next
        #         q_next does not perform backpropagation, so you need to use .detach()
        #         (Check https://pytorch.org/docs/stable/autograd.html#torch.Tensor.detach)
        # line 3: calculate the target values using and save them in q_target
        #         you can use the max(1)[0] to calculate the maximum value in each row in q_next
        #         you may need to use the .view() method to adjust the dimension of the tensor
        
        # Q-learning core steps

        # Line 1: Q(s, a) using gather to select Q-values of taken actions
        q_eval = self.q_net(b_s).gather(1, b_a)

        # Line 2: Q(s', a') using target network, no gradient tracking
        q_next = self.target_net(b_s_).detach()

        # Line 3: target = reward + gamma * max(Q(s', a'))
        q_target = b_r + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)

        ############################

        # compute loss
        loss = self.loss_func(q_eval, q_target)

        ############################
        # Your Code #
        # 3 lines: clear the gradient using zero_grad(), backpropagation, and update the weights

        self.optimizer.zero_grad() # clear gradients
        
        loss.backward() # compute gradients
        
        self.optimizer.step() # update weights

        ############################


def ratio_reward(next_state, env):
    """
    ratio reward
    :param next_state: [position, velocity, angle, angle velocity]
    :param env: the environment
    :return: ratio reward
    """
    r1 = r2 = float('inf')
    ############################
    # Your Code #
    # according to the document, state = [position, velocity, angle, angle velocity]
    # define the reward function as r1 + r2
    # r1 is the reward for the position, r2 is the reward for the angle
    # r1 is the ratio that the agent is close to the position threshold (get with env.x_threshold)
    # r2 is the ratio that the agent is close to the radian threshold (get with env.theta_threshold_radians)
    # A general form of r1 and r2 is (threshold - abs(value)) / threshold.

    # Get thresholds
    x_threshold = env.unwrapped.x_threshold
    theta_threshold = env.unwrapped.theta_threshold_radians

    # Extract state values
    position = next_state[0] # position dim in state vector
    angle = next_state[2] # angle dim in state vector

    # Compute shaped rewards as ratios
    r1 = (x_threshold - abs(position)) / x_threshold # position component
    r2 = (theta_threshold - abs(angle)) / theta_threshold # angle component

    ############################

    # return the sum of r1 and r2
    return r1 + r2

# Not used in the lab if extreme reward is not mentioned in the instruction.
# def extreme_reward(done, reward):
#     """
#     extreme reward. if terminated, return -10, else return reward
#     :param done:
#     :param reward:
#     :return:
#     """
#     ############################
#     # Your Code #
#     # if done, return -10, else return reward
#     # if this part is not mentioned in the instruction, you can ignore it
#
#     ############################


def test_dqn(args, env, state_dim, n_action):
    episode_rewards = []

    dqn = DQN(args.target_replace_iter, args.batch_size, args.memory_capacity,
              args.num_hidden, args.hidden_dim, args.lr, args.gamma, state_dim, n_action, args.soft_update_tau)

    for i in range(400):  # 400 episodes

        #print(f"i th episode:{i}")
        state, _ = env.reset()
        # the sum of the rewards of each episode
        total_reward = 0

        done = truncated = False
        while not (done or truncated):
            # choose action, use epsilon decay method as epsilon = epsilon / (i+1)
            action = dqn.choose_action(state, epsilon=args.epsilon / (i + 1), n_action=n_action)
            # take action and get the next state, reward, done and truncated
            next_state, reward, done, truncated, info = env.step(action)

            if args.reward_method == 'return':
                # no adjustment to the reward, just using reward
                save_reward = reward
            elif args.reward_method == 'ratio':
                save_reward = ratio_reward(next_state, env)
            # elif args.reward_method == 'extreme':
            #     # Not used if the extreme reward is not mentioned in the instruction.
            #     save_reward = extreme_reward(done, reward)
            else:
                raise ValueError('unknown reward method')

            dqn.store_transition(state, action, save_reward, next_state)  # store transition

            total_reward += reward  # update the sum of the rewards
            state = next_state

            ############################
            # Your Code #
            # start to learn when memory is full. learning one time for each step for the original batch size 60.
            # In the question about the batch size, you need to think about when changing batch size from 60 to 10, how
            # many times of learning can make the performance comparison fair.

            # If we run out of memory capacity, as in the buffer fills up, start learning
            if dqn.memory_counter >= dqn.memory_capacity:
                #dqn.learn(target_update_method=args.target_update_method)
                # for batch_size=10, do 6 updates/step so we see 60 samples per step
                
                # Part 3 (f) question (2)
                if args.batch_size == 10 and args.reward_method == 'ratio': # ensures that cases where batch size = 10 processes as if its 60 samples
                    for j in range(6):
                        # if j == 0:
                        #     print('Batch size correction repeat factor activated.')
                        dqn.learn(target_update_method=args.target_update_method)
                else: # if the batch size is not 10 and reward_method != 'ratio', continue as normal
                    dqn.learn(target_update_method=args.target_update_method)

            ############################

        # When the episode is done or truncated, print the reward sum and break the loop
        print(f"episode:{i},reward_sum:{total_reward}")
        episode_rewards.append(total_reward)

    print(episode_rewards)
    # define the file name
    file_name = f'DQN_UpdateTarget:{args.target_replace_iter}-batch:{args.batch_size}-memory:{args.memory_capacity}-' \
                f'num_hidden:{args.num_hidden}-hidden_dim:{args.hidden_dim}-lr:{args.lr}-' \
                f'reward:{args.reward_method}-target_update:{args.target_update_method}'

    # if the target update method is soft, add the tau to the file name
    if args.target_update_method == 'soft':
        file_name += f'-tau:{args.soft_update_tau}.png'
    else:
        file_name += '.png'
    # plot the episode rewards and average rewards
    plot_episode_rewards(episode_rewards, file_name)

