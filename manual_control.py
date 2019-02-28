import sys
import random
from utils import flag_parser
from utils import misc_util

import episode



def main():
    print('Starting.')

    args = flag_parser.parse_arguments()

    # Seed sources of randomness.
    random.seed(args.seed)

    scene = 'FloorPlan{}_physics'.format( args.scenes )
    gpu_id = 0


    # Start a new episode.
    total_reward = 0
    ep = episode.Episode(args, gpu_id, 0)
    ep.new_episode(args, scene)
    total_reward = 0
    
    while True:
        print("Reward so far: %s" %(total_reward))
        for i, action in enumerate(ep.actions_list):
            print("%s: %s" %(i, action["action"]))
        print("Choice?")

        choice = misc_util.getch()
        print()

        try:
            selection = int(choice)
            if selection < len(ep.actions_list):
                reward, terminal, success = ep.step(selection)
                total_reward += reward
                if terminal:
                    if ep.success:
                        print("Episode was successful!")
                    else:
                        print("Episode failed.")
                    print("Final reward: ", total_reward) 
                    break
                if not success:
                    print("Action failed!")
                    print()
            else:
                raise ValueError("Invalid choice")
        except ValueError as e:
            print("Invalid action: %s" %(selection))

    print("Replaying...")
    ep.slow_replay()
    print("Done.")
            

if __name__ == '__main__':
    main()
