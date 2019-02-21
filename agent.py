""" Base class for all A3C Agents. """
from __future__ import division

import torch
import random #TODO: This random is not seeded
import torch.nn.functional as F
from torch.autograd import Variable
from model import ModelInput
from utils.net_util import gpuify, resnet_input_transform
from episode import Episode

class A3CAgent:
    """ Base class for all actor-critic agents. """
    def __init__(self, model, args, rank, gpu_id=-1):
        self.gpu_id = gpu_id
        self._model = None
        self.model = model(args)
        self._episode = Episode(args, gpu_id, rank)
        self.eps_len = 0
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.done = True
        self.info = None
        self.reward = 0
        self.hidden = None
        self.actions = []
        self.verbose = args.verbose
        self.max_episode_length = args.max_episode_length
        self.hidden_state_sz = args.hidden_state_sz
        self.action_space = args.action_space

    def sync_with_shared(self, shared_model):
        """ Sync with the shared model. """
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.model.load_state_dict(shared_model.state_dict())
        else:
            self.model.load_state_dict(shared_model.state_dict())

    def eval_at_state(self):
        model_input = ModelInput()
        model_input.state = self.preprocess_frame(self.episode.state_for_agent())
        model_input.hidden = self.hidden
        model_output = self.model.forward(model_input)
        return model_output

    @property
    def state(self):
        return self.preprocess_frame(self.episode.state_for_agent())

    @property
    def episode(self):
        """ Return the current episode. """
        return self._episode

    @property
    def environment(self):
        """ Return the current environmnet. """
        return self.episode.environment

    @property
    def model(self):
        """ Returns the model. """
        return self._model

    def print_info(self):
        """ Print the actions. """
        for action in self.actions:
            print(action)

    @model.setter
    def model(self, model_to_set):
        self._model = model_to_set
        if self.gpu_id >= 0 and self._model is not None:
            with torch.cuda.device(self.gpu_id):
                self._model = self.model.cuda()

    def _increment_episode_length(self):
        self.eps_len += 1
        if self.eps_len >= self.max_episode_length:
            if not self.done:
                self.max_length = True
                self.done = True
            else:
                self.max_length = False
        else:
            self.max_length = False

    def action(self, training=True):
        """ Train the agent. """
        if training:
            self.model.train()
        else:
            self.model.eval()

        model_output = self.eval_at_state()
        self.hidden = model_output.hidden

        # Convert policy logit into probability.
        prob = F.softmax(model_output.policy, dim=1)

        if training:
            # Sample the action.
            action = prob.multinomial(1).data
        else:
            # Take the best action.
            action = prob.argmax(dim=1, keepdim=True).data

        log_prob = F.log_softmax(model_output.policy, dim=1)
        entropy = -(log_prob * prob).sum(1)
        log_prob = log_prob.gather(1, Variable(action))

        self.reward, self.done, self.info = self.episode.step(action[0, 0])

        self.entropies.append(entropy)
        self.values.append(model_output.value)
        self.log_probs.append(log_prob)
        self.rewards.append(self.reward)
        self.actions.append(action)

        self._increment_episode_length()

        # populate the success
        self.success = self.episode.success

        return model_output.value, prob, action

    
    def reset_hidden(self):
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.hidden = (torch.zeros(1, self.hidden_state_sz).cuda(), torch.zeros(1, self.hidden_state_sz).cuda())
        else:
            self.hidden = (torch.zeros(1, self.hidden_state_sz), torch.zeros(1, self.hidden_state_sz))

    def repackage_hidden(self):
        self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())

    def clear_actions(self):
        """ Clear the information stored by the agent. """
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.actions = []
        self.reward = 0

    def preprocess_frame(self, frame):
        """ Preprocess the current frame for input into the model. """
        frame = resnet_input_transform(frame, 84)
        state = torch.Tensor(frame)
        return gpuify(state.unsqueeze(0), self.gpu_id)

    def exit(self):
        self.episode.environment.controller.stop()

    def reset_episode(self):
        """ Reset the episode. """
        return self._episode.reset()
