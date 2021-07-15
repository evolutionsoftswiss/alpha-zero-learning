package ch.evolutionsoft.rl;

import java.util.Collections;
import java.util.List;
import java.util.Set;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;

public abstract class Game {

  /**
   * The value corresponds currently also to the channel
   * of the player field stones in TicTacToe
   */
  public static final int MAX_PLAYER = 1;

  /**
   * The value corresponds currently also to the channel
   * of the player field stones in TicTacToe
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
   * Can be used to track action probabilities to specific board actions during
   * the training progress.
   * 
   * @param computationGraph
   */
  public void evaluateBoardActionExamples(ComputationGraph computationGraph) {
    
  }

  /**
   * Returns the symmetries for a given board.
   * 
   * If you don't want to generate additional symmetry examples from an training example,
   * you can also return an empty list here.
   * 
   * @param board
   * @param actionProbabilities
   * @param currentPlayer
   * @param iteration
   * @return
   */
  public List<AdversaryTrainingExample> getSymmetries(INDArray board, INDArray actionProbabilities, int currentPlayer,
      int iteration) {
    
    return Collections.emptyList();
  }
  
  /**
   * Here boardSize can be used for som games, boardsize + 1 for Go with additional pass move, other
   * number of possible actions would be present for chess
   * 
   * @return number of all available moves corresponding to the number of action probabilites output
   * number of the neural net
   */
  public abstract int getNumberOfAllAvailableMoves();
  
  /**
   * For some games also identical to boardSize, boardsize + 1 for Go with additional pass move
   * after some number of stones present threshold
   * 
   * @return a number smaller or equal than getNumberOfAllAvailableMoves()
   */
  public abstract int getNumberOfCurrentMoves();

  /**
   * Return the initial board state, that is used as neural net input.
   * 
   * @return the initial board
   */
  public abstract INDArray getInitialBoard();

  /**
   * Method to make a first move with the given index.
   * This is used to get several variations of games during
   * comparison of different neural nets.
   * 
   * @return Board after first move
   */
  public abstract INDArray doFirstMove(int gameNumber);
  
  /**
   * Returns indices of valid move fields with the given board
   * 
   * @param board
   * @return a set of indices of valid move fields
   */
  public abstract Set<Integer> getValidMoveIndices(INDArray board);

  /**
   * Returns a INDArray mask containing the valid moves at a given board state.
   * This INDArray is used to mask out invalid moves with a probability > zero.
   * 
   * @param currentBoard
   * @return INDArray with valid move indices ones, zero for other indices
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
   * @param player
   * @return
   */
  public abstract INDArray makeMove(INDArray board, int moveIndex, int player);

  /**
   * 
   * @param currentBoard
   * @param player
   * @return
   */
  public abstract boolean hasWon(INDArray currentBoard, int player);
}
