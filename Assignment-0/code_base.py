from matplotlib import animation
import matplotlib.pyplot as plt
import gymnasium as gym
from get_args import get_args

def initialize_env(args):
    ##################################################
    # Part II Environment Initialization #
    # You can modify map size in get_args.py file#
    env = gym.make(args.env_name, desc=None, map_name=args.map_size, render_mode=args.render_mode, is_slippery=False)
    ##################################################


    return env

def test_action(args):
    env = initialize_env(args)
    #Run the env
    ##################################################
    # Part III Environment Functions#
    state, _ = env.reset()
    print(state)
    for action in range(env.action_space.n):
        state, _ = env.reset()
        next_state, reward, done, _, prob = env.step(action)
        print(f'action:{action}, next state:{next_state}, reward:{reward}, done:{done}, prob:{prob}')
    ##################################################


def test_moves(args):
    env = initialize_env(args)
    total_reward = 0
    num_steps = 0
    ##################################################
    # Part IV Environment Test  #
    # You can modify action list in get_args.py file#
    state, _ = env.reset()
    for i in args.actions:
        next_state, reward, done, _, _ = env.step(i)
        print(f'next state:{next_state}, reward:{reward}, done:{done}')
        total_reward = total_reward + reward
        num_steps += 1
    ##################################################

    print(f'total reward:{total_reward}, num_steps: {num_steps}')


if __name__ == "__main__":
    # get arguments from get_args.py
    args = get_args()
    print(f'mode: {args.mode}')
    # Part II
    if args.mode == 'initialize_env':
        env = initialize_env(args)
        state, _ = env.reset()
        for t in range(1000):
            action = env.action_space.sample()
            _, _, done, _, _ = env.step(action)
            if done:
                break
        env.close()
    # Part III
    elif args.mode == 'test_action':
        test_action(args)
    # Part IV
    elif args.mode == 'test_moves':
        test_moves(args)
