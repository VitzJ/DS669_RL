import gymnasium as gym


def test_each_action(env):
    """
    reset the env and test each action
    :param env: cliff walking environment
    :return:
    """
    state, _ = env.reset()
    print(state)
    for action in range(env.action_space.n):
        state, _ = env.reset()
        next_state, reward, done, _, _ = env.step(action)
        print(f'state:{state}, action:{action}, reward:{reward}, done:{done}, next_state:{next_state}')


def test_moves(env, actions):
    """
    reset the env and test the policy
    :param env: cliff walking environment
    :param actions: a list of actions to act on the environment
    :return:
    """
    total_reward = 0
    ############################
    # Your Code #
    # Imitate the test_each_action function, take the action one by one and move to the destination state #
    # You need to call the env.step to get the next state, reward, and other information #
    # Please print the state, reward, and done for each step #
    
    #• First, reset the environment.
    state, _ = env.reset()
    #print(state)

    for action in actions:
        #• First, reset the environment.
        #state, _ = env.reset()
        #• Then, take the action in the parameter actions one by one for each step. 
        next_state, reward, done, _, _ = env.step(action)
        
        #• Print the action, next state, reward, and done for each step.
        #print(f'state:{current_state}, action:{action}, reward:{reward}, done:{done}, next_state:{next_state}')
        print(f'action:{action}, next_state:{next_state}, reward:{reward}, done:{done}')

        # add the current reward to the total reward
        total_reward += reward
        # change the current state to be equal to the next state for the next action comparison
        current_state = next_state
    
    print(f'Start State: {state}')
    print(f'End State: {next_state}')

    ############################
    print(f'total reward:{total_reward}')

