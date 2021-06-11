package ch.evolutionsoft.rl4j;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import ch.evolutionsoft.net.game.tictactoe.TicTacToeConstants;
import ch.evolutionsoft.rl4j.tictactoe.TicTacToe;

public class MonteCarloSearch {

  double cUct = 1.0;
  
  int numberOfSimulations = 50;

  ComputationGraph computationGraph;
  
  TreeNode rootNode;
  TreeNode treeNode;
  
  public MonteCarloSearch(ComputationGraph computationGraph) {
    
    this(computationGraph, TicTacToeConstants.EMPTY_CONVOLUTIONAL_PLAYGROUND);
  }
  
  public MonteCarloSearch(ComputationGraph computationGraph, INDArray currentBoard) {
    
    this.computationGraph = computationGraph;
    int currentPlayer = TicTacToe.getOtherPlayer(TicTacToe.getEmptyFields(currentBoard));
    this.rootNode = new TreeNode(-1, currentPlayer, 0, 1.0f, null);
  }
  
  void playout(INDArray board) {
    
    INDArray currentBoard = board.dup();
    
    while (!treeNode.children.isEmpty()) {
      
      treeNode = treeNode.selectMove(this.cUct);
      currentBoard = TicTacToe.makeMove(currentBoard, treeNode.move, treeNode.lastColorMove);
    }
    

    INDArray oneBatchBoard = Nd4j.zeros(1, 3, 3, 3);
    oneBatchBoard.putRow(0, currentBoard);

    INDArray[] nnOutput = this.computationGraph.output(oneBatchBoard);      
    INDArray actionProbabilities = nnOutput[0];
    double leafValue = nnOutput[1].getDouble(0);
    
    if (TicTacToe.gameEnded(currentBoard)) {
      
      leafValue = 0.5f;
      
      if (TicTacToe.hasWon(currentBoard, treeNode.lastColorMove)) {
        
        leafValue = 1f;
      
      // Not possible in TicTacToe, connect four, chess and so an, but in Go
      } else if (TicTacToe.hasWon(currentBoard, TicTacToe.getOtherColor(treeNode.lastColorMove))) {
        
        leafValue = 0f;
      }
    
    } else {
      
      treeNode.expand(actionProbabilities, currentBoard);
    }

    treeNode.updateRecursiv(leafValue);
  }

  public INDArray getActionValues(INDArray board, double temperature) {
    
    for (int simulationNumber = 1; simulationNumber <= numberOfSimulations; simulationNumber++) {

      this.treeNode = rootNode;
      this.playout(board.dup());
    }

    int[] visitedCounts = new int[TicTacToeConstants.COLUMN_COUNT];
    int maxVisitedCounts = 0;

    for (int index = 0; index < TicTacToeConstants.COLUMN_COUNT; index++) {
      
      if (this.rootNode.children.containsKey(index)) {
        
        visitedCounts[index] = this.rootNode.children.get(index).timesVisited;
        if (visitedCounts[index] > maxVisitedCounts) {
          
          maxVisitedCounts = visitedCounts[index];
        }
      
      } else {
        
        visitedCounts[index] = 0;
      }
    }
    
    INDArray moveProbabilities = Nd4j.zeros(TicTacToeConstants.COLUMN_COUNT);

    if (0 == temperature) {
      
      INDArray visitedCountsArray = Nd4j.createFromArray(visitedCounts);
      
      INDArray visitedCountsMax = visitedCountsArray.argMax(0);// Lower index is taken on equal max values
      
      moveProbabilities.putScalar(visitedCountsMax.getInt(0), 1);
      
      return moveProbabilities;
    }
    
    for (int index = 0; index < TicTacToeConstants.COLUMN_COUNT; index++) {

      double softmaxProbability = 0;
      
      if (maxVisitedCounts == visitedCounts[index]) {
        
        softmaxProbability = 1;
      
      } else if (visitedCounts[index] > 0) {

        softmaxProbability = Math.exp((1 / temperature) * Math.log(visitedCounts[index]) - maxVisitedCounts);
      }
      
      moveProbabilities.putScalar(index, softmaxProbability);
    }
    
    return moveProbabilities;
  }
  
  TreeNode updateWithMove(int lastMove) {
    
    if (this.rootNode.children.containsKey(lastMove)) {
      
      this.rootNode = this.rootNode.children.get(lastMove);
      return this.rootNode;
    }
    else {
      
      throw new IllegalArgumentException("no lastMove here.");
    }
  }
}
