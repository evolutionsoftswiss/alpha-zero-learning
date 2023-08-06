package ch.evolutionsoft.rl.alphazero.connectfour.model;

/**
 * @author evolutionsoft
 */
public class PlaygroundConstants {

  public static final int EMPTY = 2;
  public static final int YELLOW = 0;
  public static final int RED = 1;
  public static final int GREY = 3;
  
  protected static final int[] columnsPrioritySorted = new int[]{3, 2, 4, 1, 5, 0, 6};
  
  public static final int COLUMN_COUNT = 7;
  public static final int ROW_COUNT = 6;
  
  // TODO Increased array size with border fields is now only used in tests
  public static final int ARRAY_COLUMN_COUNT = 9;
  public static final int ARRAY_ROW_COUNT = 8;
  
  private PlaygroundConstants() {
    // Hide constructor
  }
}
