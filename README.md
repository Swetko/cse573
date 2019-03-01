# CSE 573 Thor project #

You can do this project either individually or in groups of 2.

## 1. Installation and basic setup ##
  In this project, you will be exploring deep reinforcement learning using the [AI2-Thor](https://ai2thor.allenai.org) environment. It consists of household scenes in which various objects are scattered around. In the current problem setting, the main task is to navigate and find an object in a kitchen scene through reinforcement learning. We have currently implemented the reinforcement learning setup using an actor-critic algorithm called [Asynchronous Advantage Actor-Critic (A3C)](https://arxiv.org/pdf/1602.01783.pdf). In this part, the goal is to familiarize yourself with different components of the code and different training settings. In the second part, which will be released shortly, you will be modifying the code to apply it to new problem settings in the same environment.

  In order to speed things up, you will be using a GPU.<sup>[1](#footnote1)</sup> We've provided instructions for setting up the project on a Google Cloud Platform server in [INSTALL](INSTALL.md). If you have access to your own GPU, you may use that instead, but the setup instructions will likely be a bit different depending on the machine you use; You can contact TAs for help with the setup. Moreover if you are running on your own machine and don't have an Nvidia you might be able to run THOR. You just need OpenGL drivers. Just specify `--gpu -1` to disable training on GPU.

  Follow the instructions to set up the project. Running the command at the end of the instructions should take around an hour. You should see the training success curve approach or hit the ceiling at 1.0, while the test success rate will be quite low. You will use these results in the next question.

  `--num-test-episodes NUMBER`  specify how many episodes to test on (1000 default)

  `--num-train-episodes NUMBER`  specify how many episodes to train on (1000 default)

  `--load-model PATH` specify which model to use as initialization for training (or for testing).

  `--resume` uses the last model as the initial model (useful for resuming training).

  The training script autosaves the model every 100 steps. Also make sure the the structure of the model is exactly the same otherwise you are not going to be able to load it. So if you make changes to the Model you cannot use old saves.

  `main.py` is just for training. To test you should use `main-test.py --load-model PATH`.

## DELIVERABLES ##
Write a short report (~1 pages not including graphs) about your findings that should also contain succinct answers to the questions posed below. Deadline: March 1st (Friday), 23:59pm.

## 2. Explore the effects of training settings (20 points) ##
Now that everything is set up, you're going to explore how adjusting the training settings affects the model's performance. For each of the comparisons listed below you should provide the following:
  - Plots of training success rate vs steps for each run.
  - The difference in test success rate between runs.
  - The difference in number of training steps in each run.
  - Explain why the training adjustment results in the changes (or lack thereof) listed above.

### 2.1 Number of training scenes ###
Vary the number of scenes the model trains over. You can do this by changing the `--scenes` flag between 1 and 3.

**Note:** Training with 3 scenes will likely take 5+ hours.

### 2.2 Random initializations ###
Vary how the code initializes the scene for each training episode. You can do this by including the `--randomize-objects` flag. As the name implies, this will randomly shuffle object locations in the scene at the beginning of each episode. Does randomization help with Training/Test success rates. Why?

**Note:** Training with this flag on 1 scene will likely take an additional 1-2 hours.

### 2.3 Same testing and training scene ###
Figure out how to change so that training and testing scenes are the same. Now run training and testing and report the differences with 2.2. Note: testing on the same scene is not exactly the same as running training. Explain what is the difference.

## 3. Explore the model (20 points) ##
In this question you'll be familiarizing yourself with the code and exploring how model complexity affects performance.

### 3.1 Explain the code ###
At a high level, we need to learn a policy which, given some state, outputs a distribution over all possible actions. As this is deep RL, we'll use a deep neural network to turn the observed state into the requisite action distribution. In one or two sentences each, explain the purpose of the following parts of the code.
  - What does `a3c_loss` in `train.py` do?
  - Explain `model.py`
  - Explain the training loop in `train.py`
  - Explain the `action` function in `agent.py`

### 3.2 Explore the effect of memory (LSTM) ###
The model currently employs a recursive Neural Network. Explore `model.py`, `train.py` and `agent.py` and find out how it is implemented. Remove the recurrent connection as well as the LSTM cell and run training and testing on 1 scene with random initializations. Report the findings similar to question 2.

# Part 2 #

Now that you have seen how the agent is able to find a single object, it is time to move on to bigger and better things; finding *two* objects. In the previous part, the agent searched the scene for a tomato. In this part you will be modifying the code so that the agent learns to locate not only the tomato, but a bowl for it as well.

## Deliverables ##
- Poster outlining your approach to this problem, as well as results you obtained.
- Demo of your model running. We'll ask to see this at the poster session. You can either run it live, or show a pre-recorded video.
- A zip file containing any modified files necessary to run your model.


## 1. Finding multiple objects (40 points) ##
Your first goal is to modify the code so that it is able to successfully locate two objects: the tomato and the bowl. More explicitly, a successful episode is defined as an episode in which each of the target objects is `visible` to the agent at some point during the episode. Note that `visible` is a property of objects in the scene that depends on both the distance from the agent, as well as the agent's field of view. See the `judge` function in `episode.py` for an example of how to access this property.

You can achieve the task in any way you want. You might want to hardcode the number of objects you are looking for or you can make it variable and propagate the number throughout your code. Below are some suggestions and notes based on the TAs' experience with this problem. We strongly recommend developing on the simplest configuration (1 scene and no randomization). We will only evaluate using your training settings, so make sure the agent is able to successfully complete the simple training environment.

In order to make sure the agent consciously finds the objects you might want to add actions to indicate that an object is found (similar to how currently the `Done` action is used). To change the set of actions you have to look in several files. `constants.py` sets the action names. You also have to change the `action-space` parameter in `utils/flag_parser.py`. In `episode.py` in the function `judge` you have to implement the logic of the actions (What is success criteria, when does the agent see reward, etc.). In general those will be different than the current implementation.

The target (the tomato for Part 1) is currently specified in the initialization phase as `episode.target`. You might want to change that or to make it a vector. An episode contains the information for one run of the agent. You might want to add memory to the agent (for example to remember what objects it has seen in the past). This is essentially augmenting the state space for the the task. One place to do that is to change the `episode` class. Another might be the `agent` class. In either case make sure it is reinitialized at each episode start, otherwise your agent could think it has seen an object when in fact it had seen it many epochs ago.

Finally, if you do modify the state space, you will probably want to connect the newly augmented state to the agent's actions. You might want to change the neural network structure and add connections between the memory part of the state and actions (quite like a human). For example, one way of achieving this might be to combine the current scene embedding with the augmented state information via a linear function as follows:
```
class Model(torch.nn.Module):
    def __init__(self, args):
        ...
	self.augmented_linear = nn.Linear(additional_state_size, augmented_hidden_size)
	self.augmented_combination = nn.Linear(1024 + augmented_hidden_size, 1024)
	...

    def embedding(self, state):
        ...
	x = x.view(x.size(0), -1)
	additional_score = self.augmented_linear(additional_state_info)
	augmented_x = self.augmented_combination(torch.cat([x, additional_score]))
	return augmented_x
```
where `additional_state_info` is a 2D tensor with shape `(1, additional_state_size)`, and `augmented_hidden_size` is an integer hyperparameter.

Make sure you are able to follow the flow of information and deliver the information from the environment to the `Model`. The relevant functions to look into are `ac3lstm`, `forward` and the model constructor in `model.py`, and `state_for_agent` in `episode.py`. **NOTE: make sure you do not cheat by connecting the information from the environment (ie object locations or other properties) directly to the actions**. We want to make a realistic simulation and part of the job of an RL designer is how to ensure a realistic environment.

### Debugging ###
We’ve added a couple of potentially useful tools for debugging. The first is a manually controlled agent. You can run this using the following command.
```
vglrun python3 manual_control.py --scenes 1
```
the `--scenes` flag adjusts which scene you run on, and can be set in the range [1, 3].

We’ve also added a method in `Episode` called `slow_replay`. This will replay an episode with a delay between actions allowing you to see what the model does. You can see this in action at the end of `manual_control.py`.

## 2. Improvement (20 points) ##
Once you have a model that learns to find multiple objects, the next step is to improve it in some way. This can be in terms of the number of steps for the model to converge during training, the average reward earned, or final success rate. Most likely, the simplest way to improve the model will be to tinker with the reward function, possibly adding additional intermediate rewards, or adjusting the values, but you are free to change other aspects of the model as well.

You should show what improvement you made, and be prepared to explain how you achieved it. If you are unable to improve your model from the previous step, then explain what you tried, what you thought it would improve, and why you think it didn’t work.

## Extra Credit: Cooking the Tomato (15 points) ##
After going to all that trouble learning to find both the bowl and tomato, we've now decided that we would prefer a cooked tomato to a raw one. In order to achieve this, we'll be using the microwave. The goal for this problem is to first locate the tomato (which you are likely an expert at by this point), then pick it up, place it in the microwave, and close the microwave.

You will have to add at least 4 additional actions to the model: OpenObject, CloseObject, PickupObject, and PutObject. Each of these is supported by the Thor controller, and you can find the syntax for each [here](https://ai2thor.allenai.org/tutorials/actions). However, the executable you downloaded in Part 1 does not support the PutObject action, so you will have to replace it with an updated version that can be downloaded [here](https://courses.cs.washington.edu/courses/cse573/19wi/project/builds2.tar.gz). Once you've downloaded it, unpack it, and replace the existing `datasets/builds` directory with it.

<sub><a name="footnote1">1</a>: You may be able to use your own machine as well, running on cpu by specifying `--gpu -1`, but this will likely be slower. To do this, you will need to install Thor and download the executable from the setup script.</sub>
