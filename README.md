# CSE 573 Thor project #

## 1. Installation and basic setup ##
  In this project, you will be exploring deep reinforcement learning using AI2's Thor environment. It consists of household scenes in which various objects are scattered around. The project is split up into two parts. In this first part you will be familiarizing yourself with the code and exploring how changing the training setup affects the results. In the second part, which will be released shortly, you will be modifying the code to apply it to new problem settings in the same environment.

  In order to speed things up, you will be using a GPU. We've provided instructions for setting up the project on a Google Cloud Platform server in [INSTALL](INSTALL.md). If you have access to your own GPU, you may use that instead, but the setup instructions we provide may not work.

  Follow the instructions to set up the project. Running the command at the end of the instructions should take around an hour. You should see the training success curve approach or hit the ceiling at 1.0, while the test success rate will be quite low. You will use these results in the next question.
  
## 2. Explore the effects of training settings ##
Now that everything is set up, you're going to explore how adjusting the training settings affects the model's performance. For each of the comparisons listed below you should answer the following questions:
  - How does the training success rate change?
  - How does the test success rate change?
  - How does the training time change?
  - Explain why the training adjustment results in the changes (or lack thereof) listed above.

#### Number of training scenes ####
Vary the number of scenes the model trains over. You can do this by changing the `--scenes` flag between 1 and 3.

#### Random initializations ####
Vary how the code initializes the scene for each training episode. You can do this by including the `--randomize-objects` flag. As the name implies, this will randomly shuffle object locations in the scene at the beginning of each episode.

## 3. Explore the model ##
In this question you'll be familiarizing youself with the code and exploring how model complexity affects performance. 
  - Explain the code (a3c_loss, agent, model)

#### Linear Model ####
#### Remove LSTM ####


### 3.2 explore the effect of memory (LSTM) ###
  - Similar to above, except ablating the inclusion of the LSTM (most likely we add a flag, but we could ask them to implement it)


## Part 2 coming soon. ##

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