import argparse
import agents


def list_agents():
    for a in agents.__all__:
        print(a)


def _get_args():
    """Parse arguments from the command line and return them."""
    parser = argparse.ArgumentParser(description=__doc__)
    # add the argument for the Super Mario Bros environment to run
    parser.add_argument('--rom', '-r',
                        type=str,
                        help='The path to the ROM to play.'
                        )
    # add the argument for the mode of execution as either human or random
    parser.add_argument('--mode', '-m',
                        type=str,
                        default='human',
                        choices=['human', 'random'],
                        help='The execution mode for the emulation.',
                        )
    # add the argument for the number of steps to take in random mode
    parser.add_argument('--steps', '-s',
                        type=int,
                        default=500,
                        help='The number of random steps to take.',
                        )

    # name of agent
    parser.add_argument('--agentName', '-a',
                        type=str,
                        default='RandomAgent',
                        help='The name of the agent to be used'
                        )

    # agent configuration file path
    parser.add_argument('--agentConfig', '-ac',
                        type=str,
                        help='The location of an agent configuration file'
                        )

    # list agents
    parser.add_argument('--agentList', '-al',
                        action='store_true',
                        help='Print all agent names'
                        )

    # save state path
    parser.add_argument('--savePath', '-ss',
                        type=str,
                        help='Saves the ram upon exiting to the following path'
                        )

    # load state path
    parser.add_argument('--loadState', '-ls',
                        type=str,
                        help='Loads the RAM from the following path'
                        )

    return parser.parse_args()


def main():
    args = _get_args()
    if args.agentList:
        list_agents()
        return

    if args.agentName:
        agentName = args.agentName
        if agentName in agents.__all__:
            mod = getattr(agents, agentName)
            AC = getattr(mod, agentName)
            AC(args)
        else:
            raise Exception("Invalid agent name: " + agentName)
