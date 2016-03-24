# multiAgents.py
# --------------
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

from __future__ import division
from util import manhattanDistance
from game import Directions
import random, util


from game import Agent

def helper(newPos,list1):
        result = []
        i = 0
        while i < len(list1):
                a = manhattanDistance(newPos,list1[i])
                result.append(a)
                i = i + 1
        return result

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        numFood = successorGameState.getNumFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        ghostPositions = successorGameState.getGhostPositions()

        "*** YOUR CODE HERE ***"

        #print type(ghostState), dir(ghostState)
        totalScaredTimes = reduce(lambda x,y: x+y , newScaredTimes)
        foodDistances = helper(newPos,newFood.asList())
        capsuleDistances = helper(newPos,successorGameState.getCapsules())
        ghostDistances = helper(newPos,ghostPositions)     
        if numFood is 0:
            foodUtility = 1000 
        else:
            foodUtility = (1/numFood)
        distanceToClosestFood = 1
        distanceToClosestGhost = 1
        distanceToClosestCapsule = 1       
        if (foodDistances and min(foodDistances) != 0):
            distanceToClosestFood = min(foodDistances)              
        if (ghostDistances and min(ghostDistances) != 0):
            distanceToClosestGhost = min(ghostDistances)                    
        if (capsuleDistances and min(capsuleDistances) == 0):
            distanceToClosestCapsule = min(capsuleDistances)
        arg11 = 1/distanceToClosestFood - 1/distanceToClosestGhost
        arg22 = successorGameState.getScore() + totalScaredTimes + 1/distanceToClosestCapsule
        result = arg11 + arg22
        return result

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def maxValue(self, state, agentIndex, depth):
        if not state.isLose() and not depth==0 and not state.isWin():
            value = float("-inf")
            actionlist = state.getLegalActions(agentIndex)
            i = 0
            while i < len(actionlist):
                action = actionlist[i]
                arg1 = state.generateSuccessor(agentIndex, action)
                arg2 = (agentIndex+1) % state.getNumAgents()
                value = max(value, self.minValue(arg1,arg2, depth))
                i = i + 1
        else:            
            value = self.evaluationFunction(state)
        return value
        
    def minValue(self, state, agentIndex, depth):
        isLastGhost = False
        equalAgent = (agentIndex == state.getNumAgents()-1)
        if equalAgent:
            isLastGhost = equalAgent
        if not state.isLose() and not depth==0 and not state.isWin():
            value = float("inf")
            actionlist = state.getLegalActions(agentIndex)
            i = 0
            while i < len(actionlist):
                action = actionlist[i]
                arg1 = state.generateSuccessor(agentIndex, action)
                arg2 = (agentIndex+1) % state.getNumAgents()
                if (isLastGhost):
                    value = min(value,self.maxValue(arg1, 0, depth-1))
                else:
                    value = min(value, self.minValue(arg1,arg2, depth))
                i = i + 1
        else:
            value = self.evaluationFunction(state)
        return value

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        bestchoice = "Stop"
        value = float("-inf")
        i = 0
        actlist = gameState.getLegalActions(0)
        while i <  len(actlist):
            action = actlist[i]
            arg1 = self.minValue(gameState.generateSuccessor(0, action), 1, self.depth)
            maxVal = max(value, arg1)
            if value < maxVal:
                value = maxVal
                bestchoice = action
            i = i + 1
        return bestchoice

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def maxValue(self, state, agentIndex, depth, alpha, beta):
        if not state.isLose() and not depth==0 and not state.isWin():
            value = float("-inf")
            actionlist = state.getLegalActions(agentIndex)
            i = 0
            while i < len(actionlist):
                action = actionlist[i]
                arg1 = state.generateSuccessor(agentIndex, action)
                arg2 = (agentIndex+1) % state.getNumAgents()
                value = max(value, self.minValue(arg1, arg2, depth, alpha, beta))
                if value > beta:
                    break
                alpha = max(alpha, value)
                i = i + 1
        else:            
            value = self.evaluationFunction(state)

        return value

    def minValue(self, state, agentIndex, depth, alpha, beta):      
        isLastGhost = False
        equalAgent = (agentIndex == state.getNumAgents()-1)
        if equalAgent:
            isLastGhost = True
        if not state.isLose() and not depth==0 and not state.isWin():
            value = float("inf")
            actionlist = state.getLegalActions(agentIndex)
            i = 0
            while i < len(actionlist):
                action = actionlist[i]
                arg1 = state.generateSuccessor(agentIndex, action)
                arg2 = (agentIndex+1) % state.getNumAgents()
                if (isLastGhost):
                    value = min(value, self.maxValue(arg1, 0, depth-1, alpha, beta))
                else:
                    value = min(value, self.minValue(arg1, arg2, depth, alpha, beta))
                if value < alpha:
                    break
                beta = min(beta, value)
                i = i + 1
        else:            
            value = self.evaluationFunction(state)

        return value

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        bestchoice = "Stop"
        value = float("-inf")  
        alpha = float("-inf")
        beta = float("inf")
        i = 0
        actlist = gameState.getLegalActions(0)
        while i <  len(actlist):
            action = actlist[i]
            arg1 = self.minValue(gameState.generateSuccessor(0, action), 1, self.depth, alpha, beta)
            maxVal = max(value, arg1)
            if maxVal > beta:
                bestchoice = action
                break
            alpha = max(alpha, maxVal)
            if maxVal > value:
                value = maxVal
                bestchoice = action
            i = i + 1
        return bestchoice

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def maxValue(self, state, agentIndex, depth):
        if not state.isLose() and not depth==0 and not state.isWin():
            value = float("-inf")
            actionlist = state.getLegalActions(agentIndex)
            i = 0
            while i < len(actionlist):
                action = actionlist[i]
                arg1 = state.generateSuccessor(agentIndex, action)
                arg2 = (agentIndex+1) % state.getNumAgents()
                value = max(value, self.expectValue(arg1, arg2, depth))
                i = i + 1
        else:            
            value = self.evaluationFunction(state)
        return value

    def expectValue(self, state, agentIndex, depth):
        isLastGhost = False
        equalAgent = (agentIndex == state.getNumAgents()-1)
        if equalAgent:
            isLastGhost = equalAgent
        if not state.isLose() and not depth==0 and not state.isWin():
            value = 0
            actCount = 0
            actionlist = state.getLegalActions(agentIndex)
            i = 0        
            while i < len(actionlist):
                action = actionlist[i]
                actCount = actCount + 1
                arg1 = state.generateSuccessor(agentIndex, action)
                arg2 = (agentIndex+1) % state.getNumAgents()
                if (isLastGhost):
                    value = value + self.maxValue(arg1, 0, depth-1)
                else:
                    value = value + self.expectValue(arg1, arg2, depth)
                i = i + 1
            if actCount == 0:
                value = 0
            else:
                value = value/actCount
        else:
            value = self.evaluationFunction(state)
        return value

    def getAction(self, gameState):
        bestchoice = "Stop"
        value = float("-inf")
        i = 0
        actlist = gameState.getLegalActions(0)
        while i <  len(actlist):
            action = actlist[i]
            arg1 = self.expectValue(gameState.generateSuccessor(0, action), 1, self.depth)
            maxVal = max(value, arg1)
            if maxVal > value:
                value = maxVal
                bestchoice = action
            i = i + 1
        return bestchoice    


def betterEvaluationFunction(currentGameState):
        successorGameState = currentGameState
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        numFood = successorGameState.getNumFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        ghostPositions = successorGameState.getGhostPositions()
        totalScaredTimes = reduce(lambda x,y: x+y , newScaredTimes)
        if numFood is 0:
            foodUtility = 1000 
        else:
            foodUtility = (1/numFood)
        
        foodDistances = helper(newPos,newFood.asList())
        capsuleDistances = helper(newPos,successorGameState.getCapsules())
        ghostDistances = helper(newPos,ghostPositions)

        distanceToClosestFood = 1
        distanceToClosestGhost = 1
        distanceToClosestCapsule = 1       
        if (foodDistances and min(foodDistances) != 0):
            distanceToClosestFood = min(foodDistances)              
        if (ghostDistances and min(ghostDistances) != 0):
            distanceToClosestGhost = min(ghostDistances)                    
        if (capsuleDistances and min(capsuleDistances) == 0):
            distanceToClosestCapsule = min(capsuleDistances)
        arg11 = 1/distanceToClosestFood - 1/distanceToClosestGhost
        arg22 = successorGameState.getScore() + totalScaredTimes + 1/distanceToClosestCapsule
        result = arg11 + arg22
        return result

# Abbreviation
better = betterEvaluationFunction
