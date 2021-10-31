package ch.evolutionsoft.rl;

import java.util.Collections;
import java.util.List;
import java.util.Set;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;

public abstract class Game {

  /**
   * Constant denoting the first or max player.
   * The value corresponds currently also to the channel
   * of the player field stones in TicTacToe
   */
  public static final int MAX_PLAYER = 1;

  /**
   * Constant denoting the second or min player.
   * The value corresponds currently also to the channel
   * of the player field stones in TicTacToe
   */
  public static final int MIN_PLAYER = 2;
  

  protected int currentPlayer = MAX_PLAYER;
  
  protected INDArray currentBoard;

  protected Game() {
    
    this(Game.MAX_PLAYER);
  }
  
  protected Game(int currentPlayer) {
    
    this.currentPlayer = currentPlayer;
    this.currentBoard = getInitialBoard();
  }

  public INDArray getCurrentBoard() {
    
    return this.currentBoard.dup();
  }
  
  public int getCurrentPlayer() {
    
    return this.currentPlayer;
  }
  
  /**
   * 
   * @return int[] array with the given Set of valid move indices.
   */
  public int[] getValidIndices(Set<Integer> validIndicesList) {
    
    int[] validIndices = new int[validIndicesList.size()];

    int allMovesSize = getNumberOfCurrentMoves();
    int validIndex = 0;
    for (int index = 0; index < allMovesSize; index++) {
      
      if (validIndicesList.contains(index)) {
        validIndices[validIndex] = index;
        validIndex++;
      }
    }
    
    return validIndices;
  }

  /**
   * 
   * @param player
   * @return
   */
  public int getOtherPlayer(int player) {
    
    if (MAX_PLAYER == player) {
      
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
  
  public Game createNewInstance() {
    
    return null;
  }

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
   * @param board the board input for neural net to create symmetries for
   * @param actionProbabilities the actionProbabilities to create symmetries for
   * @param currentPlayer the player to move next
   * @param iteration iteration as additional info
   * @return
   */
  public List<AdversaryTrainingExample> getSymmetries(INDArray board, INDArray actionProbabilities, int currentPlayer,
      int iteration) {
    
    return Collections.emptyList();
  }
  
  /**
   * Here boardSize can be used for some games, boardsize + 1 for Go with additional pass move, other
   * number of possible actions would be present for chess
   * 
   * @return number of all available moves corresponding to the number of action probabilities output
   * number of the neural net.
   */
  public abstract int getNumberOfAllAvailableMoves();
  
  /**
   * For some games also identical to boardSize, boardsize + 1 for Go with additional pass move
   * after some number of stones present threshold.
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
  public abstract INDArray doFirstMove(int moveIndex);
  
  /**
   * Returns indices of valid move fields with the current board
   * state and potentially currentPlayer.
   * 
   * @return a Set of indices of valid move fields
   */
  public abstract Set<Integer> getValidMoveIndices();

  /**
   * Returns an INDArray mask containing the valid moves at a given board state.
   * This INDArray is used to mask out invalid moves with a probability > zero.
   * 
   * @return one dimensional INDArray with valid move indices ones, zero for other indices
   */
  public abstract INDArray getValidMoves();

  /**
   * 
   * @return true if the game ended, false otherwise
   */
  public abstract boolean gameEnded();

  /**
   * Perform the given move and return the new board state.
   * 
   * @param moveIndex
   * @param player
   * @return new board after move
   */
  public abstract INDArray makeMove(int moveIndex, int player);

  /**
   * Get the end result of a game. This should be a value between 0 and 1.
   * Here the current player to move does not inverse the result.
   * 
   * the lastPlayerMove is max or min and potentially usable in go. It has no effect for TicTacToe.
   * 
   * @return a value between 0 and 1, currently 1 for max win, 0 for min win and 0.5 for a draw
   */
  public abstract double getEndResult(int lastPlayer);
}
