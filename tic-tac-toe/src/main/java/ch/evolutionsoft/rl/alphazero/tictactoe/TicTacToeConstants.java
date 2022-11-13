package ch.evolutionsoft.rl.alphazero.tictactoe;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public final class TicTacToeConstants {

  private TicTacToeConstants() {
    // Hide constructor
  }

  /**
   * TicTacToe playground in one row.
   */
  public static final int ROW_COUNT = 1;
  public static final int COLUMN_COUNT = 9;
  public static final INDArray EMPTY_PLAYGROUND = Nd4j.zeros(ROW_COUNT, COLUMN_COUNT);

  /**
   * TicTacToe playground three channel 3x3 image.
   */
  public static final int IMAGE_SIZE = 3;
  public static final int IMAGE_CHANNELS = 3;
  public static final int IMAGE_POINTS = 9;
  
  public static final int CURRENT_PLAYER_CHANNEL = 0;
  public static final int MAX_PLAYER_CHANNEL = 1;
  public static final int MIN_PLAYER_CHANNEL = 2;
  
  public static final double EMPTY_IMAGE_POINT = 0;
  public static final double OCCUPIED_IMAGE_POINT = 1;
  
  public static final INDArray ZEROS_PLAYGROUND_IMAGE = Nd4j.zeros(IMAGE_SIZE, IMAGE_SIZE);
  public static final INDArray ONES_PLAYGROUND_IMAGE = Nd4j.ones(IMAGE_SIZE, IMAGE_SIZE);
  public static final INDArray MINUS_ONES_PLAYGROUND_IMAGE = ZEROS_PLAYGROUND_IMAGE.sub(1);
  
  public static final INDArray EMPTY_CONVOLUTIONAL_PLAYGROUND = Nd4j.create(IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE);
  static {
    EMPTY_CONVOLUTIONAL_PLAYGROUND.putSlice(CURRENT_PLAYER_CHANNEL, ONES_PLAYGROUND_IMAGE);
    EMPTY_CONVOLUTIONAL_PLAYGROUND.putSlice(MAX_PLAYER_CHANNEL, ZEROS_PLAYGROUND_IMAGE);
    EMPTY_CONVOLUTIONAL_PLAYGROUND.putSlice(MIN_PLAYER_CHANNEL, ZEROS_PLAYGROUND_IMAGE);
  }

  /**
   * Non empty playground fields ("crosses or circles"), empty is zero.
   */
  public static final double MAX_PLAYER = 1.0;
  public static final double MIN_PLAYER = -1.0;
  public static final double EMPTY_FIELD_VALUE = 0.0;
  
  public static final int SMALL_CAPACITY = 10;
  public static final int MEDIUM_CAPACITY = 5000;

  public static final double NET_WIN = 1.0;
  public static final double NET_DRAW = 1.0;
  public static final double NET_LOSS = 0.0;

  public static final int DEPTH_ADVANTAGE = 1;
  public static final int MINIMAX_DRAW_VALUE = 0;
}
