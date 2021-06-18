package ch.evolutionsoft.rl4j.tictactoe;

import static ch.evolutionsoft.net.game.tictactoe.TicTacToeConstants.*;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class TicTacToeStateAction {
  
  public static final INDArray ZEROS_PLAYGROUND_IMAGE = Nd4j.zeros(1, IMAGE_SIZE, IMAGE_SIZE);
  public static final INDArray ONES_PLAYGROUND_IMAGE = Nd4j.ones(1, IMAGE_SIZE, IMAGE_SIZE);
  
  public static final INDArray EMPTY_CONVOLUTIONAL_PLAYGROUND = Nd4j.create(IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE);
  static {
    EMPTY_CONVOLUTIONAL_PLAYGROUND.putRow(PLAYER_CHANNEL, ONES_PLAYGROUND_IMAGE);
    EMPTY_CONVOLUTIONAL_PLAYGROUND.putRow(MAX_PLAYER_CHANNEL, ZEROS_PLAYGROUND_IMAGE);
    EMPTY_CONVOLUTIONAL_PLAYGROUND.putRow(MIN_PLAYER_CHANNEL, ZEROS_PLAYGROUND_IMAGE);
  }
  
  INDArray boardState;
  int actionIndex = -1;

  public TicTacToeStateAction(INDArray boardState, int actionIndex) {
    
    this.boardState = boardState.dup();
    this.actionIndex = actionIndex;
  }
  
  public boolean equals(Object other) {
    
    if ( !(other instanceof TicTacToeStateAction)) {
      
      return false;
    }
    
    TicTacToeStateAction otherTicTacToeState = (TicTacToeStateAction) other;
    
    return this.boardState.equals(otherTicTacToeState.boardState) &&
        this.actionIndex == otherTicTacToeState.actionIndex;
  }
  
  public int hashCode() {
    
    return this.boardState.hashCode() + this.actionIndex;
  }
}
