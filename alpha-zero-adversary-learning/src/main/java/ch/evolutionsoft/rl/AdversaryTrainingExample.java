package ch.evolutionsoft.rl;

import java.io.Serializable;

import org.nd4j.linalg.api.ndarray.INDArray;

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
