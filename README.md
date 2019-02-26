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
The model currently employs a recursive Neural Network. Explore `model.py`, `train.py` and `agent.py` and find out how it is implemented. Remove the recurrent connection as well as the LSTM cell and run training and testing on 1 scene with random initializations. Report the findings similar to part 2.

## Part 2 coming soon. ##

<sub><a name="footnote1">1</a>: You may be able to use your own machine as well, running on cpu by specifying `--gpu -1`, but this will likely be slower. To do this, you will need to install Thor and download the executable from the setup script.</sub>

<!---
## 4. Change reward to achieve different (better) results ##

You might want to change more than just the reward function, so it's best if you get familiar with the code.

### 4.1 Play with the relative rewards for finding simple object ###
  - Changing parameters in the current reward (step penalty, finishing reward), comparisons like above

### 4.2 Change the reward to find multiple targets ###
  - Open-ended. Change the reward so the model successfully collects all targets
  - We need to add a new flag to trigger this. Select a list of targets instead of a single one.
  - Also should either have students update the state to include history (which objects have been collected), or provide that for them.

## 5. Find and object and move to microwave ##

Use the experience with memory from from 4.
  - Similar to 4, but requires ordering.

### 5.1 Explore how you add actions ###

## 6. Weighted item collection ##
Instead of providing explicit targets, assign a score to each object. The new task is to collect the highest scoring set of K objects within some time limit.

  - If we limit it so that once the model decides to pick up an object, it's stuck with it, then the model would have to learn to make decisions about expected value of further exploration compared to selecting what it's seen given the time limit.
-->
## finding multiple objects ##

You can achieve the task in any way you want. You might want to hardcode the number of objects you are looking for or you can make it variable and propagate the number throughout your code. Below are some suggestions and notes based on the TAs' experience with this problem. We were able to make the agent find 3 objects. As usual try first on the simplest configuration (1 scene and no randomization). We do not care about testing performance in this part, so make sure the agent learns in the training environment. 

In order to make sure the agent consciously finds the objects you might want to add actions to indicate that an object is found (similar to how currently the "Done" action is used). To change the set of actions you have to look in several files. `constants.py` sets the action names. You also have to change the `action-space` parameter in `flag_parser.py`. In `episode.py` in the function `judge` you have to implement the logic of the actions (What is success criteria, when does the agent see reward, etc.). In general those will be different than the current implementation.

The target is currently specified in the initialization phase as `episode.target`. You might want to change that or to make it a vector. An episode is contains the information for one run of our agent. You might want to add memory to our agent (for example to remember what objects it has seen in the past). One place to do that is to augment the `episode` class. Another might be the `agent` class. In either case make sure it is reinitialized at each episode start, otherwise your agent could think it has seen an object when in fact it had seen it many epochs ago.

Another place that might be beneficial to change is to connect the agent's memory to inform its actions. You might want to change the neural network structure and add connection between its memory and actions (quite like a human). Note that it also has implicit memory in the form of LSTM. Make sure you are able to follow the flow of information and deliver the information from the environment to the `Model` in the form of memory. The relevant functions to look into are `ac3lstm`, `forward` and the model constructor in `model.py`, and `state_for_agent` in `episode.py`. **NOTE: make sure you do not cheat by connection the information from the environment directly to the actions**. We want to make a realistic simulation and part of the job of an RL designer is how to ensure realistic environment.
