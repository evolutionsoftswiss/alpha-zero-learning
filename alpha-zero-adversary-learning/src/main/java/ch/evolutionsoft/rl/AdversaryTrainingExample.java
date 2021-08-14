package ch.evolutionsoft.rl;

import java.io.Serializable;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * An {@link AdversaryTrainingExample} is generated by self play and consists
 * of the board state, action probabilities, currentPlayer max or min, the
 * current player value after played until end.
 * iteration is only present for debug analysis e.g. to know the recentness
 * of an {@link AdversaryTrainingExample}.
 *  
 * Two {@link AdversaryTrainingExample} are considered equals when they have 
 * the same board state. Existing {@link AdversaryTrainingExample} with a given
 * board state can get replaced with new action probabilities in {@link AdversaryLearning}.
 * 
 * The currentPlayer here is the player with the next move. The currentPlayerValue is
 * 1 if the last trainingExample path lead to a win for currentPlayer 
 * Game.MAX_PLAYER or Game.MIN_PLAYER. The currentPlayerValue alternates from 1 to 0 and
 * back along {@link AdversaryTrainingExample} from alternating players.
 * An end result max win 1 or min win 0 comes from one same self play with
 * an end result not a draw. In the case of a draw currentPlayerValue is 0.5 for all examples.
 * 
 * @author evolutionsoft
 */
public class AdversaryTrainingExample implements Serializable {

  INDArray board;
  
  int currentPlayer;
  
  INDArray actionIndexProbabilities;
  
  Float currentPlayerValue;
  
  int iteration;

  public AdversaryTrainingExample(INDArray board, int currentPlayer, 
      INDArray actionIndexProbabilities, int iteration) {
    
    this.board = board.dup();
    this.currentPlayer = currentPlayer;
    this.actionIndexProbabilities = actionIndexProbabilities;
    this.iteration = iteration;
  }

  public Float getCurrentPlayerValue() {
    return currentPlayerValue;
  }

  public void setCurrentPlayerValue(Float currentPlayerValue) {
    this.currentPlayerValue = currentPlayerValue;
  }

  public INDArray getBoard() {
    return board;
  }

  public INDArray getActionIndexProbabilities() {
    return actionIndexProbabilities;
  }

  public int getCurrentPlayer() {
    return currentPlayer;
  }
  
  public boolean equals(Object other) {
    
    if (! (other instanceof AdversaryTrainingExample) ) {
      
      return false;
    }
    
    AdversaryTrainingExample otherExample = (AdversaryTrainingExample) other;
    
    return this.board.equals(otherExample.board);
  }
  
  public int hashCode() {
    
    return this.board.hashCode();
  }
  
  public String toString() {
    
    return "TrainExample {" + 
        this.board.toString() + System.lineSeparator() +
        this.currentPlayer + System.lineSeparator() +
        this.actionIndexProbabilities.toString() + System.lineSeparator() +
        this.currentPlayerValue + System.lineSeparator() +
        "}";
  }
}
