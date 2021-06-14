package ch.evolutionsoft.rl4j;

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

  /**
   * 
   * @return
   */
  public int[] getAllIndices() {
    
    int[] validIndices = new int[getFieldCount()];

    for (int index = 0; index < validIndices.length; index++) {
      
      validIndices[index] = index;
    }
    
    return validIndices;
  }

  /**
   * For most games unavailable and empty.
   * Very simple games like TicTacToe may have labels to check for.
   * 
   * @param computationGraph
   */
  public void evaluateNetwork(ComputationGraph computationGraph) {
    
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
   * Returns indices of empty fields with the given board
   * 
   * @param board
   * @return a set of indices of empty fields
   */
  public abstract Set<Integer> getEmptyFields(INDArray board);

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
   * @param emptyFields
   * @return
   */
  public abstract int getCurrentPlayer(Set<Integer> emptyFields);

  /**
   * 
   * @param currentBoard
   * @param maxPlayerChannel
   * @return
   */
  public abstract boolean hasWon(INDArray currentBoard, int maxPlayerChannel);

  /**
   * 
   * @param emptyFields
   * @return
   */
  public abstract int getOtherPlayer(Set<Integer> emptyFields);

  /**
   * 
   * @param lastColorMove
   * @return
   */
  public abstract int getOtherPlayer(int lastColorMove);

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
   * @param currentBoard
   * @return
   */
  public abstract INDArray getValidMoves(INDArray currentBoard);

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
