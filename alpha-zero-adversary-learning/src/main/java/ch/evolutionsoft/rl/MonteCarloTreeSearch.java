package ch.evolutionsoft.rl;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MonteCarloTreeSearch {
  
  Logger logger = LoggerFactory.getLogger(MonteCarloTreeSearch.class);

  double currentUctConstant = 1.5;
  
  int numberOfSimulations;

  ComputationGraph computationGraph;
  
  TreeNode rootNode;
  
  Map<INDArray, INDArray[]> neuralNetOutputsByBoardInputs = new HashMap<>();
  
  public MonteCarloTreeSearch(ComputationGraph computationGraph, AdversaryLearningConfiguration configuration) {
    
    this.computationGraph = computationGraph;
    this.currentUctConstant = configuration.getuctConstantFactor();
    this.numberOfSimulations = configuration.getNumberOfMonteCarloSimulations();
  }
  
  void playout(TreeNode treeNode, Game game) {
    
    while (treeNode.isExpanded()) {
      
      treeNode = treeNode.selectMove(this.currentUctConstant);
      game.makeMove(treeNode.lastMove, treeNode.lastMoveColor);
    }
    
    INDArray currentBoard = game.getCurrentBoard();
      
    long[] newShape = new long[currentBoard.shape().length + 1];
    System.arraycopy(currentBoard.shape(), 0, newShape, 1, currentBoard.shape().length);
    newShape[0] = 1;
    INDArray oneBatchBoard = currentBoard.reshape(newShape);
    INDArray[] neuralNetOutput = this.computationGraph.output(oneBatchBoard);
    
    INDArray actionProbabilities = neuralNetOutput[0];
    double leafValue = neuralNetOutput[1].getDouble(0);

    INDArray validActionProbabilities = actionProbabilities.mul(game.getValidMoves());
    validActionProbabilities = validActionProbabilities.div(validActionProbabilities.sumNumber());
    
    boolean gameEnded = game.gameEnded();
    if (gameEnded) {

      double endResult = game.getEndResult(treeNode.lastMoveColor);

      leafValue = endResult;
      if (Game.MAX_PLAYER == treeNode.lastMoveColor) {

        leafValue = 1 - leafValue;
      }
    }
    
    treeNode.updateRecursiv(1 - leafValue);
    
    if (!gameEnded) {

      treeNode.expand(game, validActionProbabilities);
    }
  }

  public INDArray getActionValues(Game currentGame, double temperature) {
    
    int playouts = 0;
    this.rootNode = new TreeNode(-1, currentGame.getOtherPlayer(currentGame.currentPlayer), 0, 1.0, 0.5, null);

    while (playouts < numberOfSimulations) {

      TreeNode treeNode = rootNode;
      Game newGameInstance = currentGame.createNewInstance();
      this.playout(treeNode, newGameInstance);
      playouts++;
    }
    
    int[] visitedCounts = new int[currentGame.getNumberOfCurrentMoves()];
    int maxVisitedCounts = 0;

    for (int index = 0; index < currentGame.getNumberOfCurrentMoves(); index++) {
      
      if (this.rootNode.containsChildMoveIndex(index)) {
        
        visitedCounts[index] = this.rootNode.getChildWithMoveIndex(index).timesVisited;
        if (visitedCounts[index] > maxVisitedCounts) {
          
          maxVisitedCounts = visitedCounts[index];
        }
      }
    }
    
    INDArray moveProbabilities = Nd4j.zeros(currentGame.getNumberOfCurrentMoves());

    if (0 == temperature) {
      
      INDArray visitedCountsArray = Nd4j.createFromArray(visitedCounts);
  
      INDArray visitedCountsMaximums = Nd4j.where(visitedCountsArray.gte(visitedCountsArray.amax(0).getNumber(0)), null, null)[0];

      visitedCountsArray.close();
      
      moveProbabilities.putScalar(
          visitedCountsMaximums.getInt(
              AdversaryLearningConstants.randomGenerator.nextInt((int) visitedCountsMaximums.length())),
          AdversaryLearningConstants.ONE);
      
      return moveProbabilities;
    }
    
    INDArray softmaxParameters = Nd4j.zeros(currentGame.getNumberOfCurrentMoves());
    Set<Integer> validMoveIndices = currentGame.getValidMoveIndices();
    for (int index : validMoveIndices) {

      softmaxParameters.putScalar(index, (1 / temperature) * Math.log(visitedCounts[index] + 1e-8));
    }

    double maxSoftmaxParameter = softmaxParameters.maxNumber().doubleValue();
    
    for (int index : validMoveIndices) {

      double softmaxParameter = softmaxParameters.getDouble(index);
      if (logger.isWarnEnabled() && Double.isInfinite(softmaxParameter)) {
        logger.warn("Infinite softmax param {} for valid move index {}, NaN results expected."
            + " Is {} and getValidMoveIndices() OK ?"
            + "getValidMoveIndices() had {}, but visitedCounts {}",
            softmaxParameter,
            index,
            currentGame,
            Arrays.toString(visitedCounts));
      }

      moveProbabilities.putScalar(index, Math.exp(softmaxParameter - maxSoftmaxParameter));
    }
    
    moveProbabilities = moveProbabilities.div(moveProbabilities.sumNumber());
    
    return moveProbabilities;
  }

  public void resetStoredOutputs() {
    
    this.neuralNetOutputsByBoardInputs.clear();
  }

  TreeNode updateWithMove(int lastMove) {
    
    if (this.rootNode.containsChildMoveIndex(lastMove)) {
      
      this.rootNode = this.rootNode.getChildWithMoveIndex(lastMove);
      return this.rootNode;
    }
    else {
      
      throw new IllegalArgumentException("no child with move " + lastMove +
          "found for current root node with last move" + this.rootNode.lastMove);
    }
  }
}
