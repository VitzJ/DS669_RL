import numpy as np
from codebase_train import epsilon_greedy
from util import plot_return


def q_learning(env, num_episode, gamma, alpha, init_epsilon):
    """
    Q learning algorithm
    :param env: environment
    :param num_episode: number of episodes generate from the environment
    :param gamma: gamma for decreasing the reward
    :param alpha: alpha in q learning algorithm to update q table
    :param init_epsilon: initial epsilon to control the convergence of iteration
    :return: q table for each state and action pair
    """
    # tabular_q: initialize q table
    tabular_q = np.zeros(shape=(48, 4))
    # tabular_q = np.ones(shape=(48, 4))*-1
    # total_rewards: initialize total reward list for plot
    total_rewards = []
    for episode in range(num_episode):
        # done: terminal signal, true for reaching terminal state
        done = False
        # total_rewards: initialize total reward list for plot
        total_reward = 0
        # episode_len: length of the episode
        episode_len = 0

        # state: initialize state from reset
        state, _ = env.reset()

        ############################
        # Your Code #
        # your epsilon optimization method. Copy your implementation in sarsa to here#
        epsilon = init_epsilon

        ############################

        ############################
        # Your Code
        # Implement the q learning algorithm
        # You need to call the epsilon_greedy function in train.py #
        # You need to call the env.step to get the next state, reward, and other information #
        # Do not forget to update the total reward and episode_len #
        # Please use while loop to finish this part. #



        ############################

        print("Episode:", episode, "Episode Length: ", episode_len, "Total Reward: ", total_reward)
        total_rewards.append(total_reward)

    plot_return(total_rewards, f'q_learning_alpha_{alpha}', total_reward=total_reward)
    return tabular_q
