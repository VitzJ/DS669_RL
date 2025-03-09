### Temp Warning Suppression because its annoying
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*env.nrow.*")
warnings.filterwarnings("ignore", message=".*env.ncol.*")
warnings.filterwarnings("ignore", message=".*env.P.*")
### DELETE ^^^^^^^^^^^ AFTER FINISHED WITH ASSIGNMENT

### MDP Value Iteration and Policy Iteration
import random

from get_args import get_args
import numpy as np
import gymnasium as gym
import time

np.set_printoptions(linewidth=np.inf)

np.set_printoptions(precision=3)


def interpret_policy(policy, nrow, ncol):
    '''

    interpret a 2-D policy from number to action using: 0: L 1: D 2: R 3: U
    :param policy: generated policy by a method
    :param nrow: number of rows
    :param ncol: number of columns
    :return: re_policy: policy of each state with action's first letter
    '''
    policy = policy.reshape(nrow, ncol)
    re_policy = np.zeros((nrow, ncol), dtype=str)
    for i in range(len(policy)):
        for j in range(len(policy[i])):
            if policy[i][j] == 0:
                re_policy[i][j] = 'L'
            elif policy[i][j] == 1:
                re_policy[i][j] = 'D'
            elif policy[i][j] == 2:
                re_policy[i][j] = 'R'
            elif policy[i][j] == 3:
                re_policy[i][j] = 'U'
    return re_policy


def policy_evaluation(P, nS, policy, gamma=0.9, epsilon=1e-3):
    """

    Evaluate the value function from a given policy.
    :param P: transition probability
    :param nS: number of states
    :param policy: the policy to be evaluated
    :param gamma: gamma parameter used in policy evaluation
    :param epsilon: epsilon parameter used in policy evaluation
    :return: value_function: value function from policy evaluation
             evalution_steps: the number of steps need for policy evaluation
    """

    ############################
    # Your Code #
    # Modify the following line for initialization optimization in question 5.(f)
    # Hint: Please add a new parameter for the policy_iteration function and use this parameter to control the initialization.

    # Initialize value function as all zeros
    value_function = np.zeros(nS)

    # Test 1: Initialize value function as all negative 1
    #value_function = np.ones(nS)
    #value_function = value_function * -1

    # Test 2: Copy the final value function from the default zeroes initialization, test epsilon = 0.001, 0.01, 0.1, 0.5, 1, 10
    value_function = np.array([0.254, 0.282, 0.314, 0.349, 0.387, 0.43,  0.478, 0.531, 0.282, 0.314, 0.349, 0.387, 0.43,  0.478, 0.531, 0.59,  0.314, 0.349, 0.387, 0.,    0.478, 0.531, 0.59,  0.656, 0.349, 0.387, 0.43,  0.478, 0.531, 0.,    0.656, 0.729, 0.314, 0.349, 0.387, 0.,    0.59,  0.656, 0.729, 0.81,  0.282, 0.,    0.,    0.59,  0.656, 0.729, 0.,    0.9,   0.314, 0.,    0.478, 0.531, 0.,    0.81,  0.,    1.,    0.349, 0.387, 0.43,  0.,    0.81,  0.9,   1.,    0.   ])

    #print(f'value_function with ndim {len(value_function)} :', value_function)
    ############################

    # evaluation_steps: the number of steps needed for policy evaluation in each iteration
    evaluation_steps = 0

    #print('param P:', P)
    #print('param nS:', nS)
    #print('param policy:', policy)
    #print('param gamma:', gamma)
    #print('param epsilon:', epsilon)

    ############################
    # Your Code #
    # Please use np.linalg.norm(x, np.inf) to calculate the infinity norm. #
    # Please use while loop to finish this part. #
    # Remember to update the evaluation_steps. #

    # Synchronous Backup Implementation Steps:
    # 1. Save a copy of old values: Copy the current value function to `value_function_prev` before each iteration
    # 2. Iterate over all states: Compute new values uniformly based on `value_function_prev` to avoid immediate updates affecting other states in the current iteration
    # 3. Convergence criterion: Calculate the infinity norm (max absolute difference) between old and new value functions. Terminate if below `epsilon`

    while True:
        # 1. Save a copy of old values to `value_function_prev`
        value_function_prev = np.copy(value_function)

        # 2. Iterate over all states in nS to compute new values based on `value_function_prev`
        for state in range(nS):
            action = policy[state]
            new_value = 0

            for transition in P[state][action]:
                prob, next_state, reward, done = transition
                new_value += prob * (reward + gamma * value_function_prev[next_state] * (not done))
            
            value_function[state] = new_value

        # Check if the value_function is still 1D
        if value_function.ndim != 1:
            raise ValueError(f"Expected value_function to be 1D, but got {value_function.ndim}D.")

        evaluation_steps += 1
        # 3. Convergence criterion, terminate if below epsilon
        if np.linalg.norm(value_function - value_function_prev, np.inf) <= epsilon:
            break
            
    print(f'value_function: {np.equal(value_function,value_function_prev)}')
    #print(f'Evaluation Steps: {evaluation_steps}')
    ############################

    return value_function, evaluation_steps


def policy_improvement(P, nS, nA, value_function, gamma=0.9):
    """
    Use the value function to improve the policy.
    :param P: transition probability
    :param nS: number of states
    :param nA: number of actions
    :param value_function: value function from policy iteration
    :param gamma: gamma parameter used in policy improvement
    :return: new_policy: An array of integers. Each integer is the optimal action to take in that state according to
                the environment dynamics and the given value function.
    """

    new_policy = np.zeros(nS, dtype="int")

    ############################
    # Your Code #
    # Please use np.argmax to select the best actions after getting the q value of each action. #

    #print('\nparam P:', P)
    #print('param nS:', nS)
    #print('param nA:', nA)
    #print('param value_function:', value_function)
    #print('new policy:', new_policy)
    #print('param gamma:', gamma)

    # Ensure that value_function is a 1D array
    if value_function.ndim != 1:
        raise ValueError(f"value_function should be a 1D array, but found {value_function.ndim} dimensions")

    # initialize the `new_policy` array
    new_policy = np.zeros(nS, dtype="int")

    for state in range(nS):
        # Initialize the `q_value` array
        q_values = np.zeros(nA)

        # Get the `q_value` of each action
        for action in range(nA):
            for transition in P[state][action]:
                #print(transition)
                prob, next_state, reward, done = transition

                #print(f"\nState: {state}, Action: {action}")
                #print(f"Transition: (prob={prob}, next_state={next_state}, reward={reward}, done={done})")
                #print(f"value_function[{next_state}] = {value_function[next_state]}")

                # Ensure value_function[next_state] is a scalar
                assert np.isscalar(value_function[next_state]), f"Non-scalar value detected: {value_function[next_state]}"

                q_values[action] += prob * (reward + gamma * value_function[next_state] * (not done))
        
        # Use `np.argmax(q_values)` to select the best actions
        new_policy[state] = np.argmax(q_values)

    ############################
    return new_policy


def policy_iteration(P, nS, nA, init_action=-1, gamma=0.9, epsilon=1e-3):
    """
    Runs policy iteration. Please call the policy_evaluation() and policy_improvement() methods to implement this method.
    :param P: transition probability
    :param nS: number of states
    :param nA: number of actions
    :param init_action: initial action for all the states, -1 for random action
    :param gamma: gamma parameter used in policy_evaluation() and policy_improvement()
    :param epsilon: epsilon parameter used in policy_evaluation()
    :return: value_function: np.ndarray[nS]
	         policy: np.ndarray[nS]
	         iteration: int, the number of iterations needed for policy iteration
    """

    value_function = np.zeros(nS)

    ############################
    # Your Code #
    # for the question of policy iteration initialization optimization #
    # Initialize policy #
    init_policy = np.random.randint(0, nA, nS) if init_action == -1 else np.ones(nS, dtype=int) * init_action
    ############################

    # Number of iterations. The iteration does not include the steps of policy evaluation.
    iteration = 0

    # previous policy: the policy of last iteration.
    policy_prev = init_policy.copy()

    ############################
    # Your Code #
    # Please call the policy_evaluation() and policy_improvement() to update the policy. #
    # Remember to update the iteration and policy_prev. #
    # Please use while loop to finish this part. #
    # The time complexity of the code within the while loop represents the running time required "in one iteration" as mentioned in II.(c)#

    while True:
        iteration += 1
        value_function, evaluation_steps = policy_evaluation(P, nS, policy_prev, gamma, epsilon)
        new_policy = policy_improvement(P, nS, nA, value_function, gamma)

        print(f'Evaluation Steps in Iteration {iteration}: {evaluation_steps}')
        #print(f'Value Function: {value_function}\n')

        if np.array_equal(policy_prev, new_policy):
            break

        policy_prev = new_policy
    
    policy = policy_prev
    ############################

    print(f"There are {iteration} iterations in policy iteration.")
    return value_function, policy, iteration


def value_iteration(P, nS, nA, init_value=0.0, gamma=0.9, epsilon=1e-3):
    """
    Learn value function and policy by using value iteration method for a given gamma and environment.
    :param P: transition probability
    :param nS: number of states
    :param nA: number of actions
    :param init_value: initial value for value iteration
    :param gamma: gamma parameter used in value_iteration()
    :param epsilon: epsilon parameter used in value_iteration()
    :return: value_function: np.ndarray[nS]
	         policy: np.ndarray[nS]
	         iteration: int, the number of iterations needed for value iteration
    """


    # Initialize value #
    value_function = np.ones(nS) * init_value

    # policy: the policy output from the generated value function after value iteration.
    policy = np.zeros(nS, dtype=int)
    iteration = 0

    ############################
    # Your Code #
    # Please use np.argmax to select the best action after getting the q value of each action. #
    # Please use np.linalg.norm(x, np.inf) to calculate the infinity norm. #
    # Please use while loop to finish this part. #
    # The time complexity of the code within the while loop represents the running time required "in one iteration" as mentioned in II.(d)#

    while True:
        iteration += 1
        value_function_prev = value_function.copy()
        #delta = 0

        for state in range(nS):
            q_values = np.zeros(nA)

            for action in range(nA):

                for transition in P[state][action]:
                    #print(transition)
                    prob, next_state, reward, done = transition
                    #print(f"\nState: {state}, Action: {action}")
                    #print(f"Transition: (prob={prob}, next_state={next_state}, reward={reward}, done={done})")
                    #print(f"value_function[{next_state}] = {value_function[next_state]}")

                    q_values[action] += prob * (reward + gamma * value_function[next_state] * (not done))

            best_action_value = np.max(q_values)
            #delta = max(delta, abs(value_function[state] - best_action_value))

            value_function[state] = best_action_value

            policy[state] = np.argmax(q_values)
        
        # Convergence criterion, terminate if below epsilon
        delta = np.linalg.norm(value_function - value_function_prev, np.inf)

        if delta <= epsilon:
            break

        #if delta < epsilon:
            #break

    ############################

    # uncomment the following line if you need to print the value function
    # print('value_function:', value_function)

    print(f"There are {iteration} iterations in value iteration.")
    return value_function, policy, iteration


def render_single(env, policy, max_steps=100):
    '''

    This function does not need to be modified
    Renders policy once on environment. Watch your agent play!

    :param env: gym.core.Environment. Environment to play on. Must have nS, nA, and P as attributes.
    :param policy: np.array of shape [env.nS]. The action to take at a given state
    :param max_steps: the maximum number of iterations
    :return: None
    '''

    episode_reward = 0
    state, _ = env.reset()
    for t in range(max_steps):
        env.render()
        #time.sleep(0.25)
        action = policy[state]
        state, reward, done, _, _ = env.step(action)
        episode_reward += reward
        if done:
            break
    env.render()
    if not done:
        print(f"The agent didn't reach a terminal state in {max_steps} steps.")
    else:
        print(f"Episode reward: {episode_reward}")


# Edit below to run policy and value iteration on different environments and
# visualize the resulting policies of actions.

if __name__ == "__main__":
    # get arguments from get_args.py
    args = get_args()

    # Initialize the gym environment and render
    env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", render_mode=args.render_mode, is_slippery=False)
    # Please check this link for the definition of state and actions of the FrozenLake game:
    # https://www.gymlibrary.dev/environments/toy\_text/frozen\_lake/
    
    # Number of state is 8 * 8 = 64
    env.nS = env.nrow * env.ncol
    # Number of action is 4
    env.nA = 4

    # Uncomment the following line to check and understand the format of the transition probability of FrozenLake.
    #print('transition probability:', env.P)

    # Running time start point
    start = time.time()
    # Initialize the average iteration
    avg_iteration = 0

    # Run the algorithm for "args.seeds" times. Each time with a different random seed.
    for i in range(args.seeds):

        # Reset the environment
        env.reset()
        # Set the random seed
        np.random.seed(i)
        random.seed(i)

        if args.method == 'policy_iteration':
            # Run policy iteration
            print("---- Policy Iteration----\n")
            value, policy, iteration = policy_iteration(
                env.P, env.nS, env.nA, init_action=args.init_action, gamma=args.gamma, epsilon=args.epsilon)

        elif args.method == 'value_iteration':
            # Run value iteration
            print("---- Value Iteration----\n")
            value, policy, iteration = value_iteration(
                env.P, env.nS, env.nA, init_value=args.init_value, gamma=args.gamma, epsilon=args.epsilon)
        else:
            raise ValueError('Unknown method')
        # Cumulate the number of iterations
        avg_iteration += iteration

        # Print the policy, interpreted to the actions' first letter (check the interpret_policy function).
        print('policy:', interpret_policy(policy.reshape(env.nrow, env.ncol), env.nrow, env.ncol))

    print('Total running time is:', time.time() - start,
          ' average number of iteration is:', avg_iteration / args.seeds)

    # Render the policy, the rendering do not require screenshots.
    render_single(env, policy, 100)
