package ch.evolutionsoft.rl;

import java.util.HashMap;
import java.util.Map;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ch.evolutionsoft.net.game.NeuralNetConstants;

public class MonteCarloSearch {
  
  Logger logger = LoggerFactory.getLogger(MonteCarloSearch.class);

  double cUct = 1.5;
  
  int numberOfSimulations;
  
  Game game;

  ComputationGraph computationGraph;
  
  TreeNode rootNode;
  
  Map<INDArray, INDArray[]> neuralNetOutputsByBoardInputs = new HashMap<INDArray, INDArray[]>();
  
  public MonteCarloSearch(Game game, ComputationGraph computationGraph, AdversaryLearningConfiguration configuration) {
    
    this(game, computationGraph, configuration, game.getInitialBoard());
  }
  
  public MonteCarloSearch(Game game, ComputationGraph computationGraph, AdversaryLearningConfiguration configuration, INDArray currentBoard) {
    
    this.game = game;
    this.computationGraph = computationGraph;
    this.rootNode = new TreeNode(-1, game.getOtherPlayer(game.currentPlayer), 0, 1.0, 0.5, null);
    this.cUct = configuration.getCpUct();
    this.numberOfSimulations = configuration.getNumberOfMonteCarloSimulations();
  }
  
  void playout(TreeNode treeNode, Game game, INDArray board) {
    
    INDArray currentBoard = board.dup();
    
    while (treeNode.isExpanded()) {
      
      treeNode = treeNode.selectMove(this.cUct);
      currentBoard = game.makeMove(currentBoard, treeNode.move, treeNode.lastColorMove);
    }
      
    long[] newShape = new long[currentBoard.shape().length + 1];
    System.arraycopy(currentBoard.shape(), 0, newShape, 1, currentBoard.shape().length);
    newShape[0] = 1;
    INDArray oneBatchBoard = currentBoard.reshape(newShape);
    INDArray[] neuralNetOutput = this.computationGraph.output(oneBatchBoard);
    
    INDArray actionProbabilities = neuralNetOutput[0];
    double leafValue = neuralNetOutput[1].getDouble(0);
    
    if (game.gameEnded(currentBoard)) {

      double endResult = game.getEndResult(currentBoard, treeNode.lastColorMove);

      leafValue = endResult;
      if (Game.MIN_PLAYER == treeNode.lastColorMove) {
      
        leafValue = 1 - leafValue;
      }
      
    } else {

      treeNode.expand(game, actionProbabilities, currentBoard);
    }

    treeNode.updateRecursiv(leafValue);
  }

  public INDArray getActionValues(INDArray board, double temperature) {
    
    int playouts = 0;

    while (playouts < numberOfSimulations) {

      TreeNode treeNode = rootNode;
      Game newGameInstance = game.createNewInstance();
      this.playout(treeNode, newGameInstance, board.dup());
      playouts++;
    }
    
    int[] visitedCounts = new int[game.getNumberOfCurrentMoves()];
    int maxVisitedCounts = 0;

    for (int index = 0; index < game.getNumberOfCurrentMoves(); index++) {
      
      if (this.rootNode.containsChildMoveIndex(index)) {
        
        visitedCounts[index] = this.rootNode.getChildWithMoveIndex(index).timesVisited;
        if (visitedCounts[index] > maxVisitedCounts) {
          
          maxVisitedCounts = visitedCounts[index];
        }
      }
    }
    
    INDArray moveProbabilities = Nd4j.zeros(game.getNumberOfCurrentMoves());

    if (0 == temperature) {
      
      INDArray visitedCountsArray = Nd4j.createFromArray(visitedCounts);
  
      INDArray visitedCountsMaximums = Nd4j.where(visitedCountsArray.gte(visitedCountsArray.amax(0).getNumber(0)), null, null)[0];
      
      moveProbabilities.putScalar(
          visitedCountsMaximums.getInt(
              NeuralNetConstants.randomGenerator.nextInt((int) visitedCountsMaximums.length())),
          NeuralNetConstants.ONE);
      
      return moveProbabilities;
    }
    
    INDArray softmaxParameters = Nd4j.zeros(game.getNumberOfCurrentMoves());
    for (int index : game.getValidMoveIndices(board)) {

      softmaxParameters.putScalar(index, (1 / temperature) * Math.log(visitedCounts[index]) + 1e-8);
    }

    double maxSoftmaxParameter = softmaxParameters.maxNumber().doubleValue();
    
    for (int index : game.getValidMoveIndices(board)) {

      moveProbabilities.putScalar(index, Math.exp(softmaxParameters.getDouble(index) - maxSoftmaxParameter)); 
    }
    
    moveProbabilities = moveProbabilities.div(moveProbabilities.sumNumber());
    
    return moveProbabilities;
  }

  public void resetStoredOutputs() {
    
    this.neuralNetOutputsByBoardInputs.clear();
  }
  
  public double getcUct() {
    return cUct;
  }

  public void setcUct(double cUct) {
    this.cUct = cUct;
  }

  public int getNumberOfSimulations() {
    return numberOfSimulations;
  }

  public void setNumberOfSimulations(int numberOfSimulations) {
    this.numberOfSimulations = numberOfSimulations;
  }

  TreeNode updateWithMove(int lastMove) {
    
    if (this.rootNode.containsChildMoveIndex(lastMove)) {
      
      this.rootNode = this.rootNode.getChildWithMoveIndex(lastMove);
      return this.rootNode;
    }
    else {
      
      throw new IllegalArgumentException("no lastMove here: " + lastMove);
    }
  }
}
