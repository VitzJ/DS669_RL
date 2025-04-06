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

        epsilon = init_epsilon / (episode + 1) # optimized implementation from sarsa part II b.

        ############################

        ############################
        # Your Code
        # Implement the q learning algorithm
        # You need to call the epsilon_greedy function in train.py #
        # You need to call the env.step to get the next state, reward, and other information #
        # Do not forget to update the total reward and episode_len #
        # Please use while loop to finish this part. #

        while not done:
            # Choose action using epsilon-greedy policy
            action = epsilon_greedy(tabular_q, state, epsilon)

            # Take action, get next state, reward, and done flag
            next_state, reward, done, _, _ = env.step(action)

            # Q-learning update rule
            best_next_action = np.argmax(tabular_q[next_state])
            td_target = reward + gamma * tabular_q[next_state][best_next_action]
            td_error = td_target - tabular_q[state][action]
            tabular_q[state][action] += alpha * td_error

            # Update tracking variables
            total_reward += reward
            episode_len += 1
            state = next_state

        ############################

        print("Episode:", episode, "Episode Length: ", episode_len, "Total Reward: ", total_reward)
        total_rewards.append(total_reward)

    plot_return(total_rewards, f'q_learning_alpha_{alpha}', total_reward=total_reward)
    return tabular_q
