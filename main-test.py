from __future__ import print_function, division

import os
import random
import ctypes
import setproctitle
import json
import time
import copy

import numpy as np
import torch
import torch.multiprocessing as mp

from utils import flag_parser
from utils.class_finder import optimizer_class
from utils.net_util import ScalarMeanTracker

import model
import agent
import train

os.environ["OMP_NUM_THREADS"] = "1"


def main():
    print('Starting.')

    setproctitle.setproctitle('A3C Manager')
    args = flag_parser.parse_arguments()

    create_shared_model = model.Model
    init_agent = agent.A3CAgent
    optimizer_type = optimizer_class(args.optimizer)

    start_time = time.time()
    local_start_time_str = \
        time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime(start_time))

    # Seed sources of randomness.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.enable_logging:
        from tensorboardX import SummaryWriter
        log_dir = 'runs/' + args.title + '-' + local_start_time_str
        log_writer = SummaryWriter(log_dir=log_dir)

    if args.gpu_ids == -1:
        args.gpu_ids = [-1]
    else:
        torch.cuda.manual_seed(args.seed)
        mp.set_start_method('spawn', force=True)

    print('=> Creating the shared model and optimizer.')
    shared_model = create_shared_model(args)

    shared_model.share_memory()

    if (args.resume):
        shared_model.load_state_dict(torch.load('./models/last_model'))
    elif (args.load_model != ''):
        shared_model.load_state_dict(torch.load(args.load_model))
    else:
        print("NO MODEL SUPPLIED")
        return

    print('=> Creating the agents.')
    processes = []

    end_flag = mp.Value(ctypes.c_bool, False)

    ## TEST ##
    if (args.num_test_episodes == 0):
        return
    print("Testing...")
    # Turn on random initialization for testing
    args.randomize_objects = True
    end_flag.value = False
    test_res_queue = mp.Queue()
    for rank in range(0, args.workers):
        p = mp.Process(target=train.test, args=(
            rank, args, create_shared_model,
            shared_model, init_agent,
            test_res_queue, end_flag))
        p.start()
        processes.append(p)
        print('* Agent created.')
        time.sleep(0.1)

    test_total_ep = 0
    n_frames = 0

    test_thin = args.test_thin
    test_scalars = ScalarMeanTracker()

    try:
        while test_total_ep < args.num_test_episodes:
            test_result = test_res_queue.get()
            test_scalars.add_scalars(test_result)
            test_total_ep += 1
            n_frames += test_result["ep_length"]
            if args.enable_logging and test_total_ep % test_thin == 0:
                log_writer.add_scalar("n_frames", n_frames, test_total_ep)
                tracked_means = test_scalars.pop_and_reset()
                for k in tracked_means:
                    log_writer.add_scalar(
                        k + "/test", tracked_means[k], test_total_ep
                    )

    finally:
        if args.enable_logging:
            log_writer.close()
        end_flag.value = True
        for p in processes:
            time.sleep(0.1)
            p.join()


if __name__ == '__main__':
    main()
