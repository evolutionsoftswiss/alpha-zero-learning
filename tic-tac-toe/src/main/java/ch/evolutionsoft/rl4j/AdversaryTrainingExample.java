package ch.evolutionsoft.rl4j;

import java.io.Serializable;

import org.nd4j.linalg.api.ndarray.INDArray;

public class AdversaryTrainingExample implements Serializable {

  INDArray board;
  
  int currentPlayer;
  
  INDArray actionIndexProbabilities;
  
  Integer currentPlayerValue;

  public AdversaryTrainingExample(INDArray board, int currentPlayer, 
      INDArray actionIndexProbabilities) {
    
    this.board = board.dup();
    this.currentPlayer = currentPlayer;
    this.actionIndexProbabilities = actionIndexProbabilities;
  }

  public Integer getCurrentPlayerValue() {
    return currentPlayerValue;
  }

  public void setCurrentPlayerValue(Integer currentPlayerValue) {
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
}
