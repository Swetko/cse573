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
        log_dir =  'runs/' + args.title + '-' + local_start_time_str
        log_writer = SummaryWriter(log_dir=log_dir)

    if args.gpu_ids == -1:
        args.gpu_ids = [-1]
    else:
        torch.cuda.manual_seed(args.seed)
        mp.set_start_method('spawn', force=True)

    print('=> Creating the shared model and optimizer.')
    shared_model = create_shared_model(args)

    shared_model.share_memory()
    optimizer = optimizer_type(
        filter(lambda p: p.requires_grad, shared_model.parameters()), 
        args)
    optimizer.share_memory()

    if (args.resume):
        shared_model.load_state_dict(torch.load('./models/last_model'))
    elif (args.load_model!=''):
        shared_model.load_state_dict(torch.load(args.load_model))


    print('=> Creating the agents.')
    processes = []

    end_flag = mp.Value(ctypes.c_bool, False)

    train_res_queue = mp.Queue()
    for rank in range(0, args.workers):
        p = mp.Process(target=train.train, args=(
            rank, args, create_shared_model, 
            shared_model, init_agent,
            optimizer, train_res_queue, end_flag))
        p.start()
        processes.append(p)
        print('* Agent created.')
        time.sleep(0.1)

    train_total_ep = 0
    n_frames = 0

    train_thin = args.train_thin
    train_scalars = ScalarMeanTracker()

    success_tracker = []
    
    try:
        while train_total_ep < args.num_train_episodes:
            train_result = train_res_queue.get()
            train_scalars.add_scalars(train_result)
            train_total_ep += 1
            n_frames += train_result["ep_length"]
            if train_total_ep % 100 == 0:
                torch.save(shared_model.state_dict(), './models/model_{}'.format(train_total_ep))
            if args.enable_logging and train_total_ep % train_thin == 0:
                log_writer.add_scalar("n_frames", n_frames, train_total_ep)
                tracked_means = train_scalars.pop_and_reset()
                for k in tracked_means:
                    log_writer.add_scalar(
                        k + "/train", tracked_means[k], train_total_ep
                    )
            success_tracker.append(train_result["success"])
            if len(success_tracker) > 100:
                success_tracker.pop(0)
            if len(success_tracker) >= 100 and sum(success_tracker) / len(success_tracker) > args.train_threshold:
                break
    finally:
        if args.enable_logging:
            log_writer.close()
        end_flag.value = True
        for p in processes:
            time.sleep(0.1)
            p.join()

    torch.save(shared_model.state_dict(), './models/last_model')


if __name__ == '__main__':
    main()
