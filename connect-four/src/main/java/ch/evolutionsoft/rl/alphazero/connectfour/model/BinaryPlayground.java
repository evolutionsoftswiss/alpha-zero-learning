package ch.evolutionsoft.rl.alphazero.connectfour.model;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * @author evolutionsoft
 */
public class BinaryPlayground implements Playground {

  public static final int FIRST_PLAYER = 0;
  public static final int SECOND_PLAYER = 1;
  public static final int HEIGHT = 6;
  public static final int WIDTH = 7;
  public static final int SIZE = HEIGHT * WIDTH;
  public static final int BITS_PER_COLUMN = HEIGHT + 1;
  public static final int BIT_SIZE = WIDTH * BITS_PER_COLUMN;
  public static final int RIGHT_PLACE = BITS_PER_COLUMN;
  public static final int UPPER_PLACE = 1;
  public static final int LOWER_RIGHT_PLACE = BITS_PER_COLUMN - 1;
  public static final int UPPER_RIGHT_PLACE = BITS_PER_COLUMN + 1;
  public static final int RIGHT_PLACE2 = 2 * RIGHT_PLACE;
  public static final int UPPER_PLACE2 = 2 * UPPER_PLACE;
  public static final int LOWER_RIGHT_PLACE2 = 2 * LOWER_RIGHT_PLACE;
  public static final int UPPER_RIGHT_PLACE2 = 2 * UPPER_RIGHT_PLACE;

  public static final long TOP;
  static {
    long allBitsTrue = (1L << BIT_SIZE) - 1L;
    int sevenBitsTrue = (1 << BITS_PER_COLUMN) - 1;
    TOP = (allBitsTrue / sevenBitsTrue) << HEIGHT; // has i*BITS_PER_COLUMN-th Bit true
  }

  private long[] positions;
  private int[] firstFreeBitOfColumns;
  private int[] columnHeights;
  private int fieldsLeft;
  private LineDirection fourInARowDirection = null;

  public BinaryPlayground() {

    this(new long[2], new int[7], 42);
  }

  public BinaryPlayground(BinaryPlayground playGround) {

    this.positions = new long[2];
    System.arraycopy(playGround.positions, 0, this.positions, 0, 2);

    this.columnHeights = new int[7];
    System.arraycopy(playGround.columnHeights, 0, this.columnHeights, 0, 7);

    this.firstFreeBitOfColumns = new int[7];
    System.arraycopy(playGround.firstFreeBitOfColumns, 0, this.firstFreeBitOfColumns, 0, 7);

    this.fieldsLeft = playGround.fieldsLeft;
  }

  public BinaryPlayground(long[] position, int[] heightOfColumn, int fieldsLeft) {

    this.positions = position;
    this.columnHeights = heightOfColumn;
    this.firstFreeBitOfColumns = this.makeFirstFreeBitOfColumn(heightOfColumn);
    this.fieldsLeft = fieldsLeft;
  }

  public BinaryPlayground(ArrayPosition position, int fieldsLeft) {

    this.positions = this.makePosition(position.getPosition());
    this.columnHeights = position.getColumnHeights();
    this.firstFreeBitOfColumns = this.makeFirstFreeBitOfColumn(this.columnHeights);
    this.fieldsLeft = fieldsLeft;
  }

  // --------- getters ---------------------------------------------------------

  public long getFirstPlayerPosition() {

    return this.positions[FIRST_PLAYER];
  }

  public long getSecondPlayerPosition() {

    return this.positions[SECOND_PLAYER];
  }

  public long getPosition(int player) {

    return this.positions[player];
  }

  public long[] getPosition() {

    return this.positions;
  }

  public int[] getColumnHeights() {

    return this.columnHeights;
  }

  public int getHeightOfColumn(int column) {

    return this.columnHeights[column];
  }

  public int[] getFirstFreeBitOfColumn() {

    return this.firstFreeBitOfColumns;
  }

  public int getFirstFreeBitOfColumn(int column) {

    if (column < 0 || column > 6)
      throw new IllegalArgumentException("Column out of range.");
    return this.firstFreeBitOfColumns[column];
  }

  public int getFieldsLeft() {

    return this.fieldsLeft;
  }

  public void reset() {

    this.positions = new long[2];
    this.columnHeights = new int[7];
    this.firstFreeBitOfColumns = this.makeFirstFreeBitOfColumn(this.columnHeights);
    this.fieldsLeft = 42;
    this.fourInARowDirection = null;
  }

  public int setField(int column, int color) {

    return this.trySetField(column, color);

  }

  public int setFieldEmpty(int column, int color) {

    return this.trySetFieldEmpty(column, color);

  }

  public int trySetField(int column, int color) {

    this.positions[color] ^= 1L << this.firstFreeBitOfColumns[column]++;

    this.columnHeights[column]++;

    this.fieldsLeft--;

    return this.columnHeights[column] - 1;
  }

  public int trySetFieldEmpty(int column, int color) {

    this.columnHeights[column]--;

    this.fieldsLeft++;

    this.positions[color] ^= 1L << --this.firstFreeBitOfColumns[column];

    return this.columnHeights[column] - 1;
  }

  public boolean isValidMove(int column) {
    return this.columnHeights[column] < HEIGHT;
  }

  public List<Integer> getAvailableColumns() {

    List<Integer> result = new ArrayList<>(7);

    for (int index = 0; index < 7; index++) {

      if (this.columnHeights[PlaygroundConstants.columnsPrioritySorted[index]] < 6) {

        result.add(PlaygroundConstants.columnsPrioritySorted[index]);
      }
    }

    return result;
  }

  public List<Integer> playableColumnsPriorityOrderedColumnFirst(int column) {

    List<Integer> columnsPriorityOrdered = this.getAvailableColumns();

    int indexOfColumn = columnsPriorityOrdered.indexOf(column);

    int tmp = columnsPriorityOrdered.remove(indexOfColumn);

    columnsPriorityOrdered.add(0, tmp);

    return columnsPriorityOrdered;
  }

  @Override
  public boolean fourInARow(int lastMove, int color) {

    return this.fourInARow(this.positions[color]);
  }

  public boolean fourInARow(long position) {

    long y = position & (position >> RIGHT_PLACE);
    if ((y & (y >> RIGHT_PLACE2)) != 0) { // check horizontal -
      this.fourInARowDirection = LineDirection.HORIZONTALLY;
      return true;
    }

    y = position & (position >> LOWER_RIGHT_PLACE);
    if ((y & (y >> LOWER_RIGHT_PLACE2)) != 0) { // check diagonal \
      this.fourInARowDirection = LineDirection.DIAGONALLY_DOWN;
      return true;
    }

    y = position & (position >> UPPER_RIGHT_PLACE); // check diagonal /
    if ((y & (y >> UPPER_RIGHT_PLACE2)) != 0) {
      this.fourInARowDirection = LineDirection.DIAGONALLY_UP;
      return true; 
    }

    y = position & (position >> UPPER_PLACE); // check vertical |
    this.fourInARowDirection = LineDirection.VERTICALLY;
    return (y & (y >> UPPER_PLACE2)) != 0;
  }

  public Line getWinningRow(int column, int color) {

    int index = this.getLastPlayedIndex(column);
    int row = index / PlaygroundConstants.COLUMN_COUNT;
    int forwardCount = 0;
    int backwardCount = 0;
    int indexDirection = 0;
    int rowDirection = 0;
    int columnDirection = 0;
    
    if (this.fourInARowDirection == LineDirection.VERTICALLY){

      indexDirection = UPPER_PLACE;
      rowDirection = 1;
      columnDirection = 0;
    }
    else if (this.fourInARowDirection == LineDirection.HORIZONTALLY){

      indexDirection = RIGHT_PLACE;
      rowDirection = 0;
      columnDirection = 1;
    }
    else if (this.fourInARowDirection == LineDirection.DIAGONALLY_DOWN){

      indexDirection = LOWER_RIGHT_PLACE;
      rowDirection = -1;
      columnDirection = 1;
    }
    else if (this.fourInARowDirection == LineDirection.DIAGONALLY_UP){

      indexDirection = UPPER_RIGHT_PLACE;
      rowDirection = 1;
      columnDirection = 1;
    }

    forwardCount = this.countStonesForward(row + rowDirection, column + columnDirection, color, rowDirection, columnDirection);
    backwardCount = this.countStonesBackward(row - rowDirection, column - columnDirection, color, -rowDirection, -columnDirection);
    
    if (indexDirection > 0 && forwardCount + backwardCount >= 3){
      
      int conversedDirection = convertBinaryToArrayDirection(indexDirection);
      Field beginningField = new Field(index + conversedDirection * forwardCount);
      Field endField = new Field(index - conversedDirection * backwardCount);
        
      return new Line(beginningField, endField, color);
    }
    
    return null;
  }

  public String toString() {

    StringBuilder stringBuilder = new StringBuilder();
    for (int row = HEIGHT - 1; row >= 0; row--) {
      for (int columnBit = row; columnBit < BIT_SIZE; columnBit += BITS_PER_COLUMN) {
        long mask = 1L << columnBit;
        if ((this.positions[FIRST_PLAYER] & mask) != 0)
          stringBuilder.append("X ");
        else if ((this.positions[SECOND_PLAYER] & mask) != 0)
          stringBuilder.append("O ");
        else
          stringBuilder.append(". ");
      }
      stringBuilder.append("\n");
    }
    return stringBuilder.toString();
  }

  @Override
  public boolean equals(Object otherObject) {

    if (!(otherObject instanceof BinaryPlayground)) {

      return false;
    }

    BinaryPlayground otherPlayground = (BinaryPlayground) otherObject;

    return Arrays.equals(this.positions, otherPlayground.positions);
  }

  public int hashCode() {
    long longHash = this.positions[FIRST_PLAYER] ^ this.positions[SECOND_PLAYER] * 31;
    return (int) (longHash ^ (longHash >>> 32));
  }

  public long[] makePosition(int[] arrayPosition) {

    long[] position = new long[2];

    for (int index = 0; index < arrayPosition.length; index++) {
      
      int row = (index - PlaygroundConstants.ARRAY_COLUMN_COUNT) / PlaygroundConstants.ARRAY_COLUMN_COUNT;
      int column = (index - 1) % PlaygroundConstants.ARRAY_COLUMN_COUNT;

      if (arrayPosition[index] == PlaygroundConstants.YELLOW)
        position[FIRST_PLAYER] |= 1L << (column * PlaygroundConstants.COLUMN_COUNT + row);

      else if (arrayPosition[index] == PlaygroundConstants.RED)
        position[SECOND_PLAYER] |= 1L << (column * PlaygroundConstants.COLUMN_COUNT + row);

    }

    return position;
  }

  protected int[] makeFirstFreeBitOfColumn(int[] heightOfColumn) {

    int[] firstFreeBitOfColumn = new int[7];

    for (int column = 0; column < WIDTH; column++)
      firstFreeBitOfColumn[column] = column * BITS_PER_COLUMN + heightOfColumn[column];

    return firstFreeBitOfColumn;
  }
  
  
  protected int getLastPlayedIndex(int column){
    
    return (this.columnHeights[column] - 1) * PlaygroundConstants.COLUMN_COUNT + column;
  }
  
  
  protected int countStonesForward(int row, int column, int color, int rowDirection, int columnDirection){
    
    if ((positions[color] & 1L << column * PlaygroundConstants.COLUMN_COUNT + row) == 0){
      
      return 0;
    }
    
    return 1 + countStonesForward(row + rowDirection, column + columnDirection, color, rowDirection, columnDirection);
  }


  protected int countStonesBackward(int row, int column, int color, int rowDirection, int columnDirection){
    
    if ((positions[color] & 1L << column * PlaygroundConstants.COLUMN_COUNT + row) == 0){
      
      return 0;
    }
  
    return 1 + countStonesBackward(row + rowDirection, column + columnDirection, color, rowDirection, columnDirection);
  }

  protected int convertBinaryToArrayDirection(int direction) {
   
    if (direction == UPPER_PLACE) {
 
      return PlaygroundConstants.COLUMN_COUNT;
    }
    if (direction == RIGHT_PLACE) {

      return 1;
    }
    if (direction == LOWER_RIGHT_PLACE) {
  
      return - PlaygroundConstants.COLUMN_COUNT + 1;
    }
    if (direction == UPPER_RIGHT_PLACE) {
      
      return PlaygroundConstants.COLUMN_COUNT + 1;
    }
    
    return -1;
  }
}
