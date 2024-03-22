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
                 epsilon: float = 0.5,
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

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
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
        "*** YOUR CODE HERE ***"
        reward = -0.04
        pacman_position = endState.getPacmanPosition()
        if (endState.getGhostPosition(1) == pacman_position):
            reward -= 10
        if endState.hasFood(pacman_position[0], pacman_position[1]):
            reward += 100
        return reward

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
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
        "*** YOUR CODE HERE ***"
        return self.q_table.get((state, action), 0.0)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Args:
            state: The given state

        Returns:
            q_value: the maximum estimated Q-value attainable from the state
        """
        "*** YOUR CODE HERE ***"
        legal_actions = state.getLegalPacmanActions()
        # print(state)
        if legal_actions == []:
            return 0
        if Directions.STOP in legal_actions:
            legal_actions.remove(Directions.STOP)
        # print(legal_actions)
        max_q_value = max((self.getQValue(state, action) for action in legal_actions))
        return max_q_value

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
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
        "*** YOUR CODE HERE ***"
        old_q_value = self.getQValue(state, action)
        max_future_q = self.maxQValue(nextState)
        new_q_value = old_q_value + self.alpha * (reward + max_future_q - old_q_value)
        self.q_table[(state, action)] = new_q_value

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def updateCount(self,
                    state: GameStateFeatures,
                    action: Directions):
        """
        Updates the stored visitation counts.

        Args:
            state: Starting state
            action: Action taken
        """
        "*** YOUR CODE HERE ***"
        self.counts[(state, action)] += 1
        
    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
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
        "*** YOUR CODE HERE ***"
        return self.counts.get((state, action))

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def explorationFn(self,
                      utility: float,
                      counts: int) -> float:
        """
        Computes exploration function.
        Return a value based on the counts

        HINT: Do a greed-pick or a least-pick

        Args:
            utility: expected utility for taking some action a in some given state s
            counts: counts for having taken visited

        Returns:
            The exploration value
        """
        "*** YOUR CODE HERE ***"
    
        # Compute exploration value
        exploration_value = utility / (counts + 1)
        
        return exploration_value

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getAction(self, state: GameState) -> Directions:
        """
        Choose an action to take to maximise reward while
        balancing gathering data for learning

        If you wish to use epsilon-greedy exploration, implement it in this method.
        HINT: look at pacman_utils.util.flipCoin

        Args:
            state: the current state

        Returns:
            The action to take
        """
        # The data we have about the state of the game
        # print(state)
        location = state.getPacmanPosition()
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
            
        if self.alpha != 0:
            if (self.q_table == {}):
                # print(legal)
                action = random.choice(legal)
            elif util.flipCoin(self.epsilon):
                # print(self.temp)
                exploration_values = {}
                for action in legal:
                    if (location, action) in self.q_table:
                        if self.counts.get((location, action)) < 200:
                            q_value = self.getQValue(state, action)
                            exploration_values[action] = self.explorationFn(q_value, self.getCount(location, action))
                        # print(exploration_values)
                    if exploration_values != {}:
                        action = max(exploration_values, key=exploration_values.get)
                    else:
                        action = random.choice(legal)
            else: 
                max_q_action = None
                max_q_value = float('-inf')
                for a in legal:
                    q_value = self.getQValue(location, a)
                    if q_value > max_q_value:
                        max_q_value = q_value
                        action = a
            next_state = state.generatePacmanSuccessor(action)
            if (location, action) not in self.q_table:
                self.q_table[location, action] = 0
                self.counts[location, action] = 1
                for a in legal:
                    if a != action:
                        self.counts[location, a] = 0
                        self.q_table[location, a] = 0 
            else:   
                self.updateCount(location, action)
            self.learn(location, action, self.computeReward(state, next_state), next_state)
            return action
        else :
            # print(state.getFood()[1][1])
            # print(self.q_table)
            # print(self.counts)
            max_q_action = None
            max_q_value = -1000000
            for action in legal:
                next_state = state.generatePacmanSuccessor(action)
                if (next_state.getGhostPosition(1) != next_state.getPacmanPosition()):
                    q_value = self.getQValue(location, action)
                    # print(action, q_value)
                    if q_value > max_q_value:
                        max_q_value = q_value
                        max_q_action = action
            if max_q_action is None:
                return random.choice(legal)
            return max_q_action
                

                # logging to help you understand the inputs, feel free to remove
                # print("Legal moves: ", legal)
                # print("Pacman position: ", state.getPacmanPosition())
                # print("Ghost positions:", state.getGhostPositions())
                # print("Food locations: ")
                # print(state.getFood())
                # print("Score: ", state.getScore())

                # Now pick what action to take.
                # The current code shows how to do that but just makes the choice randomly.


    def final(self, state: GameState):
        """
        Handle the end of episodes.
        This is called by the game after a win or a loss.

        Args:
            state: the final game state
        """
        # print(f"Game {self.getEpisodesSoFar()} just ended!")

        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        # print(self.q_table)
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            # print('%s\n%s' % (msg, '-' * len(msg)))
            self.setAlpha(0)
            self.setEpsilon(0)
