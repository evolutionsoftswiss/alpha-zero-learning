package ch.evolutionsoft.rl;

import java.util.List;
import java.util.Set;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;

public abstract class Game {

  /**
   * The value corresponds currently also to the channel
   * of the player field stones
   */
  public static final int MAX_PLAYER = 1;

  /**
   * The value corresponds currently also to the channel
   * of the player field stones
   */
  public static final int MIN_PLAYER = 2;
  

  protected int currentPlayer = MAX_PLAYER;
  
  public Game(int currentPlayer) {
    
    this.currentPlayer = currentPlayer;
  }

  /**
   * 
   * @return
   */
  public int[] getValidIndices(Set<Integer> validIndicesList) {
    
    int[] validIndices = new int[validIndicesList.size()];

    int allMovesSize = getNumberOfCurrentMoves();
    for (int validIndex = 0, index = 0; index < allMovesSize; index++) {
      
      if (validIndicesList.contains(index)) {
        validIndices[validIndex] = index;
        validIndex++;
      }
    }
    
    return validIndices;
  }

  /**
   * 
   * @param lastColorMove
   * @return
   */
  public int getOtherPlayer(int lastColorMove) {
    
    if (MAX_PLAYER == lastColorMove) {
      
      return MIN_PLAYER;
    }
    
    return MAX_PLAYER;
  }

  /**
   * For most games unavailable and empty.
   * Very simple games like TicTacToe may have labels to check for.
   * 
   * @param computationGraph
   */
  public void evaluateNetwork(ComputationGraph computationGraph) {
    
  }

  public Object savePosition() {
    
    return null;
  }
  
  public void restorePosition(Object savedPosition) { }
  
  /**
   * 
   * @return boardSize as default, boardsize + 1 for Go with additional pass move, other
   * number of possible actions would be present for chess
   */
  public int getNumberOfAllAvailableMoves() {
    
    return getFieldCount();
  }
  
  /**
   * 
   * @return boardSize as default, boardsize + 1 for Go with additional pass move
   * after some number of stones present threshold
   */
  public int getNumberOfCurrentMoves() {
    
    return getFieldCount();
  }
  
  /**
   * 
   * @return board size of the game
   */
  public abstract int getFieldCount();

  /**
   * Method to make a fix first move with the given index
   * 
   * @return Board after first move
   */
  public abstract INDArray doFirstMove(int index);
  
  /**
   * Returns indices of valid move fields with the given board
   * 
   * @param board
   * @return a set of indices of valid move fields
   */
  public abstract Set<Integer> getValidMoveIndices(INDArray board);

  /**
   * 
   * @param currentBoard
   * @return
   */
  public abstract INDArray getValidMoves(INDArray currentBoard);

  /**
   * 
   * @param board
   * @return true if the game ended, false otherwise
   */
  public abstract boolean gameEnded(INDArray board);

  /**
   * 
   * @param board
   * @param moveIndex
   * @param maxPlayerChannel
   * @return
   */
  public abstract INDArray makeMove(INDArray board, int moveIndex, int maxPlayerChannel);

  /**
   * 
   * @param currentBoard
   * @param maxPlayerChannel
   * @return
   */
  public abstract boolean hasWon(INDArray currentBoard, int maxPlayerChannel);

  /**
   * 
   * @return
   */
  public abstract INDArray getInitialBoard();

  /**
   * 
   * @param computationGraph
   */
  public abstract void evaluateOpeningAnswers(ComputationGraph computationGraph);

  /**
   * 
   * @param dup
   * @param dup2
   * @param currentPlayer
   * @param iteration
   * @return
   */
  public abstract List<AdversaryTrainingExample> getSymmetries(INDArray board, INDArray actionProbabilities, int currentPlayer,
      int iteration);
}
