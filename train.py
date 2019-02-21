from __future__ import division

import time
import setproctitle
import random
import os

import torch
from torch.autograd import Variable

def train(rank, args, create_shared_model, shared_model, 
          initialize_agent, optimizer, res_queue, end_flag):
    """ Training loop for each agent. """

    random.seed(args.seed + rank)
    scene = 'FloorPlan{}_physics'.format( (rank % args.scenes) + 1 )     
    setproctitle.setproctitle('Training Agent: {}'.format(rank))
    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]

    import torch
    torch.cuda.set_device(gpu_id)
    import torch.optim as optim
    from torch.autograd import Variable

    torch.manual_seed(args.seed + rank)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed + rank)

    player = initialize_agent(create_shared_model, args, rank, gpu_id=gpu_id)

    while not end_flag.value:

        # Start a new episode.
        total_reward = 0
        player.eps_len = 0
        new_episode(args, player, scene)

        while not player.done:
            # Make sure model is up to date.
            player.sync_with_shared(shared_model)
            # Run episode for num_steps or until player is done.
            for _ in range(args.num_steps):
                player.action()
                total_reward = total_reward + player.reward
                if player.done:
                    break

            # Compute the loss.
            policy_loss, value_loss = a3c_loss(args, player, gpu_id)
            total_loss = policy_loss + 0.5 * value_loss  

            loss = dict(total_loss=total_loss, policy_loss=policy_loss, value_loss=value_loss)

            # Compute gradient.
            player.model.zero_grad()
            loss['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(player.model.parameters(), 100.0)
            # Transfer gradient to shared model and step optimizer.
            transfer_gradient_from_player_to_shared(player, shared_model, gpu_id)
            optimizer.step()

            # Clear actions and repackage hidden.
            if not player.done:
                reset_player(player)
        
        # Itemize loss for logging.
        for k in loss:
            loss[k] = loss[k].item()
        
        # Log the data from the episode and reset the plyaer.
        if args.enable_logging:
            log_episode(player, res_queue, total_reward=total_reward, **loss)
            
        reset_player(player)
        
    player.exit()

def test(rank, args, create_shared_model, shared_model, 
          initialize_agent, res_queue, end_flag):
    """ Training loop for each agent. """

    random.seed(args.seed + rank)
    scene = 'FloorPlan4_physics'#.format( 4 - (rank % (4-args.scenes)))     
    setproctitle.setproctitle('Test Agent: {}'.format(rank))
    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]

    import torch
    torch.cuda.set_device(gpu_id)
    import torch.optim as optim
    from torch.autograd import Variable

    torch.manual_seed(args.seed + rank)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed + rank)

    player = initialize_agent(create_shared_model, args, rank, gpu_id=gpu_id)

    while not end_flag.value:

        # Start a new episode.
        total_reward = 0
        player.eps_len = 0
        new_episode(args, player, scene)

        while not player.done:
            # Make sure model is up to date.
            player.sync_with_shared(shared_model)
            # Run episode for num_steps or until player is done.
            for _ in range(args.num_steps):
                player.action(training=False)
                total_reward = total_reward + player.reward
                if player.done:
                    break


            # Clear actions and repackage hidden.
            if not player.done:
                reset_player(player)
        
        
        # Log the data from the episode and reset the plyaer.
        if args.enable_logging:
            log_episode(player, res_queue, total_reward=total_reward)
            
        reset_player(player)
        
    player.exit()


def new_episode(args, player, scene):
    player.episode.new_episode(args, scene)
    player.reset_hidden()
    player.done = False 

def log_episode(player, res_queue, **kwargs):
    results = {
        'ep_length': player.eps_len,
        'success': int(player.success)
    }

    results.update(**kwargs)
    res_queue.put(results)

def reset_player(player):
    player.clear_actions()
    player.repackage_hidden()

def a3c_loss(args, player, gpu_id):
    """ Evaluates the model at the current state. """
    R = torch.zeros(1, 1)
    if not player.done:
        output = player.eval_at_state()
        R = output.value.data

    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            R = R.cuda()

    player.values.append(Variable(R))
    policy_loss = 0
    value_loss = 0
    gae = torch.zeros(1, 1)
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            gae = gae.cuda()
    R = Variable(R)
    for i in reversed(range(len(player.rewards))):
        R = args.gamma * R + player.rewards[i]
        advantage = R - player.values[i]
        value_loss = value_loss + 0.5 * advantage.pow(2)

        # Generalized Advantage Estimation
        delta_t = player.rewards[i] + args.gamma * \
            player.values[i + 1].data - player.values[i].data

        gae = gae * args.gamma * args.tau + delta_t

        policy_loss = policy_loss - \
            player.log_probs[i] * \
            Variable(gae) - args.beta * player.entropies[i]

    return policy_loss, value_loss

def transfer_gradient_from_player_to_shared(player, shared_model, gpu_id):
    """ Transfer the gradient from the player's model to the shared model
        and step """
    for param, shared_param in zip(player.model.parameters(), shared_model.parameters()):
        if shared_param.requires_grad:
            if param.grad is None:
                shared_param._grad = torch.zeros(shared_param.shape)
            elif gpu_id < 0:
                shared_param._grad = param.grad
            else:
                shared_param._grad = param.grad.cpu()
