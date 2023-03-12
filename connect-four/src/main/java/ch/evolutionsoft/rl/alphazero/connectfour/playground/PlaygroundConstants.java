package ch.evolutionsoft.rl.alphazero.connectfour.playground;

/**
 * @author evolutionsoft
 */
public class PlaygroundConstants {

  public static final int EMPTY = 2;
  public static final int YELLOW = 0;
  public static final int RED = 1;
  public static final int GREY = 3;
  
  protected static final int[] columnsPrioritySorted = new int[]{3, 2, 4, 1, 5, 0, 6};

  public static final int UPPER_LEFT = 0;
  public static final int LOWER_LEFT = 63;
  public static final int UPPER_RIGHT = 8;
  public static final int LOWER_RIGHT = 71;
  
  public static final int COLUMN_COUNT = 7;
  public static final int ROW_COUNT = 6;
	
  public static final int ARRAY_COLUMN_COUNT = 9;
  public static final int ARRAY_ROW_COUNT = 8;
  
  private PlaygroundConstants() {
    // Hide constructor
  }
}
