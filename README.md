# CSE 573 Thor project #

## 1. Installation and basic setup ##
  - Basic setup instructions (starting from creating a cloud instance to setting up thor)
  
### Run basic learning example with 1 scene. ###
  - Run a simple command (ie train on a single scene without randomization) and verify that the output matches the provided output

## 2. Explore the effects of randomization ##
  - Basically the same as above, but with randomization flag on/off
  - Answer the following questions:
    - How does training performance vary?
    - How does test performance vary?
    - How does training time vary? 

## 3. Explore the model ##
  - Explain the code (a3c_loss, agent, model)

### 3.1 make sure it works for >1 scene ###
  - Compare performance (similar to above) with training on 1 scene vs training on 2 or 3 scenes

### 3.2 explore the effect of memory (LSTM) ###
  - Similar to above, except ablating the inclusion of the LSTM (most likely we add a flag, but we could ask them to implement it)

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
