# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# This template was originally adapted to KCL by Simon Parsons, but then
# revised and updated to Py3 for the 2022 course by Dylan Cope and Lin Li

from __future__ import absolute_import
from __future__ import print_function

import random
import numpy as np

from pacman import Directions, GameState
from pacman_utils.game import Agent
from pacman_utils import util


class GameStateFeatures:
    """
    Wrapper class around a game state where you can extract
    useful information for your Q-learning algorithm

    WARNING: We will use this class to test your code, but the functionality
    of this class will not be tested itself
    """

    def __init__(self, state: GameState):
        """
        Args:
            state: A given game state object
        """
        "*** YOUR CODE HERE ***"
        self.food = state.getFood()
        

class QLearnAgent(Agent):

    def __init__(self,
                 alpha: float = 0.8,
                 epsilon: float = 0.05,
                 gamma: float = 0.8,
                 maxAttempts: int = 30,
                 numTraining: int = 10):
        """
        These values are either passed from the command line (using -a alpha=0.5,...)
        or are set to the default values above.

        The given hyperparameters are suggestions and are not necessarily optimal
        so feel free to experiment with them.

        Args:
            alpha: learning rate
            epsilon: exploration rate
            gamma: discount factor
            maxAttempts: How many times to try each action in each state
            numTraining: number of training episodes
        """
        super().__init__()
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.maxAttempts = int(maxAttempts)
        self.numTraining = int(numTraining)
        # Count the number of games we have played
        self.episodesSoFar = 0
        
        # Initialise Empty Q Table and Counts libraries
        self.q_table = {}
        self.counts = {}

    # Accessor functions for the variable episodesSoFar controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value: float):
        self.epsilon = value

    def getAlpha(self) -> float:
        return self.alpha

    def setAlpha(self, value: float):
        self.alpha = value

    def getGamma(self) -> float:
        return self.gamma

    def getMaxAttempts(self) -> int:
        return self.maxAttempts


    @staticmethod
    def computeReward(startState: GameState,
                      endState: GameState) -> float:
        """
        Args:
            startState: A starting state
            endState: A resulting state

        Returns:
            The reward assigned for the given trajectory
        """
        
        # Gives initial reward of 0.04
        reward = 0.04
        
        # Gets pacman position
        pacman_position = endState.getPacmanPosition()
        
        # Checks if end state is in the same position as the ghost
        if (endState.getGhostPosition(1) == pacman_position):
            # Negative Reward
            reward -= 1
        
        # Checks if food is in the future states position
        if startState.hasFood(pacman_position[0], pacman_position[1]) == True:
            # Positive Reward
            reward += 200
            
        # Return Reward 
        return reward

    def getQValue(self,
                  state: GameStateFeatures,
                  action: Directions) -> float:
        """
        Args:
            state: A given state
            action: Proposed action to take

        Returns:
            Q(state, action)
        """
        # Return q value for state,action pair, 0.0 if it doesn't exist
        return self.q_table.get((state, action), 0.0)

    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Args:
            state: The given state

        Returns:
            q_value: the maximum estimated Q-value attainable from the state
        """
        # Gets legal actions and returns the action with the highest q value
        legal_actions = state.getLegalPacmanActions()
        if legal_actions == []:
            return 0
        if Directions.STOP in legal_actions:
            legal_actions.remove(Directions.STOP)
        max_q_value = max((self.getQValue(state, action) for action in legal_actions))
        return max_q_value

    def learn(self,
              state: GameStateFeatures,
              action: Directions,
              reward: float,
              nextState: GameStateFeatures):
        """
        Performs a Q-learning update

        Args:
            state: the initial state
            action: the action that was took
            nextState: the resulting state
            reward: the reward received on this trajectory
        """
        # Gets old q value and the max future q value
        old_q_value = self.getQValue(state, action)
        max_future_q = self.maxQValue(nextState)
        # Q value formula
        new_q_value = old_q_value + self.getAlpha() * (reward + self.getGamma() * (max_future_q - old_q_value))
        self.q_table[(state, action)] = new_q_value
        
    def updateCount(self,
                    state: GameStateFeatures,
                    action: Directions):
        """
        Updates the stored visitation counts.

        Args:
            state: Starting state
            action: Action taken
        """
        # Increments state action count
        self.counts[(state, action)] += 1
        
    def getCount(self,
                 state: GameStateFeatures,
                 action: Directions) -> int:
        """
        Args:
            state: Starting state
            action: Action taken

        Returns:
            Number of times that the action has been taken in a given state
        """
        # Return visitation counter for state action pair
        return self.counts.get((state, action))

    def explorationFn(self,
                      utility: float,
                      counts: int) -> float:
        """
        Computes exploration function.
        Return a value based on the counts

        Implementation of Least pick

        Args:
            utility: expected utility for taking some action a in some given state s
            counts: counts for having taken visited

        Returns:
            The exploration value
        """
        
        # Divides the utility by the counts
        exploration_value = utility / (counts + 1)
        
        return exploration_value

    def getAction(self, state: GameState) -> Directions:
        """
        Args:
            state: the current state

        Returns:
            The action to take
        """
        
        # Gets location of pacman and legal moves 
        location = state.getPacmanPosition()
        legal = state.getLegalPacmanActions()
        
        # Removes stop move
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
            
        # Checks if pacman is still learning
        if self.alpha != 0:
            
            # Checks if its initial move, makes a random move if q table is empty
            if (self.q_table == {}):
                action = random.choice(legal)
            
            # Flips coin to see if pacman should explore based on input epsilon
            elif util.flipCoin(self.epsilon):
                
                # Loops through legal actions and gets the exploration value for each state action
                exploration_values = {}
                for action in legal:
                    if (location, action) in self.q_table:
                        
                        # Checks if state action pair has been explored more than 10 times to encourage exploring other moves 
                        if self.counts.get((location, action)) < 10:
                            q_value = self.getQValue(state, action)
                            exploration_values[action] = self.explorationFn(q_value, self.getCount(location, action))
                    # Selects max exploration value, if empty, makes random move
                    if exploration_values != {}:
                        action = max(exploration_values, key=exploration_values.get)
                    else:
                        action = random.choice(legal)
            else: 
                # Gets max q value action
                max_q_action = None
                max_q_value = float('-inf')
                for a in legal:
                    q_value = self.getQValue(location, a)
                    if q_value > max_q_value:
                        max_q_value = q_value
                        action = a
            # Gets successor state
            next_state = state.generatePacmanSuccessor(action)
            
            # Checks if state action hasnt been explored yet
            if (location, action) not in self.q_table:
                # Initialise state action pair to q table and count table, and all other actions possible from that state
                self.q_table[location, action] = 0
                self.counts[location, action] = 1
                for a in legal:
                    if a != action:
                        self.counts[location, a] = 0
                        self.q_table[location, a] = 0
            # Else update count
            else:   
                self.updateCount(location, action)
                
            # Learn from move
            self.learn(location, action, self.computeReward(state, next_state), next_state)
            
            #Return Move
            return action
        
        # If learning is done
        else :
            # Gets highest q value move
            max_q_action = None
            max_q_value = float('-inf')
            for action in legal:
                next_state = state.generatePacmanSuccessor(action)
                if (next_state.getGhostPosition(1) != next_state.getPacmanPosition()):
                    q_value = self.getQValue(location, action)
                    if q_value > max_q_value:
                        max_q_value = q_value
                        max_q_action = action
            # Error handling for no legal moves due to ghost
            if max_q_action is None:
                max_q_action = random.choice(legal)
                
            # Returns move
            return max_q_action
                

    def final(self, state: GameState):
        """
        Handle the end of episodes.
        This is called by the game after a win or a loss.

        Args:
            state: the final game state
        """
        
        #checks if training is complete
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            
            #Deactivated training
            msg = 'Training Done (turning off epsilon and alpha)'
            self.setAlpha(0)
            self.setEpsilon(0)
