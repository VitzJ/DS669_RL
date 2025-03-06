import argparse


def get_args():
    parser = argparse.ArgumentParser()

    # Mode
    parser.add_argument('-mode', type=str, choices=['initialize_env', 'test_action', 'test_moves'],
                        default='initialize_env', help='choose the function to implement')



    # II.b Initializations
    parser.add_argument('-env_name', type=str, default='FrozenLake-v1',
                        help='environment name')
    ###############################
    # Your Code #
    # Please set the parameter 'map_size' to an appropriate value.
    parser.add_argument('-map_size', type=str, choices=['4x4', '8x8'], default='4x4',
                        help='map size')


    #IV.b action list required to create
    ###############################
    # Your Code #
    #Please set the parameter actions's default value to an appropriate list
    parser.add_argument('-actions', nargs='+', type=int, default=[2,1,2,1,1,2,2,3,2,2,1,1,0,0,1,1], help='create an action list asked in part IV.b')

    # Render mode
    parser.add_argument(
        "-render_mode",
        "-r",
        type=str,
        help="The render mode for the environment. 'human' opens a window to render. 'ansi' does not render anything.",
        choices=["human", "ansi"],
        default="human",
    )

    return parser.parse_args()



