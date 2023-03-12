package ch.evolutionsoft.rl.alphazero.connectfour.playground;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * @author evolutionsoft
 */
public class BinaryPlayground{

    
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
        TOP = (allBitsTrue / sevenBitsTrue) << HEIGHT; //has i*BITS_PER_COLUMN-th Bit true 
    }
    
    private long[] positions;
    private int[] firstFreeBitOfColumns;
    private int[] columnHeights;
    private int fieldsLeft;

    
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
    
    
    public BinaryPlayground(Object position, int fieldsLeft) {
    	
    	if (position instanceof ArrayPlayground){
    		
    		ArrayPlayground arrayPlayground = (ArrayPlayground) position;
        	
            this.positions = this.makePosition(arrayPlayground.getPosition());
            this.columnHeights = this.makeHeightOfColumn(arrayPlayground);
            this.firstFreeBitOfColumns = this.makeFirstFreeBitOfColumn(this.columnHeights);
            this.fieldsLeft = fieldsLeft;
    	}
    	else {
    		throw new IllegalArgumentException("Type " + position.getClass() + " not supported");
    	}
    }
    
    
	//--------- getters ---------------------------------------------------------
    
    
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
    
    
    //--------- setters ---------------------------------------------------------
    

    //--------- methods ---------------------------------------------------------


	public void setField(int column, int color) {

		this.trySetField(column, color);
		
	}

	
	public void setFieldEmpty(int column, int color) {

		this.trySetFieldEmpty(column, color);
		
	}
	

    public int trySetField(int column, int color) {
    	
        this.positions[color] ^= 1L << this.firstFreeBitOfColumns[column]++;
        
        this.columnHeights[column]++;
        
        this.fieldsLeft--;
        
        return this.columnHeights[column] - 1;
    }
    
    
    public void trySetFieldEmpty(int column, int color){
    	
        this.columnHeights[column]--;
        
        this.fieldsLeft++;
        
        this.positions[color] ^= 1L << --this.firstFreeBitOfColumns[column];
    }
    
    
    public boolean isPlayable(int column) {
        return this.columnHeights[column] < HEIGHT;
    }
    
    
    public List<Integer> getAvailableColumns() {

    	List<Integer> result = new ArrayList<Integer>(7);
		
		for (int index = 0; index < 7; index++){
			
			if (this.columnHeights[PlaygroundConstants.columnsPrioritySorted[index]] < 6){
				
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
    
    
    public boolean fourInARow(long position) {
    	
        long y = position & (position >> RIGHT_PLACE);
        if ((y & (y >> RIGHT_PLACE2)) != 0) // check horizontal -
          return true;
        
        y = position & (position >> LOWER_RIGHT_PLACE);
        if ((y & (y >> LOWER_RIGHT_PLACE2)) != 0) // check diagonal \
          return true;
        
        y = position & (position >> UPPER_RIGHT_PLACE); // check diagonal /
        if ((y & (y >> UPPER_RIGHT_PLACE2)) != 0)
          return true;
        
        y = position & (position >> UPPER_PLACE); // check vertical |
        return (y & (y >> UPPER_PLACE2)) != 0;
    }
    
    
    public String toString() {
        StringBuffer bufferedString = new StringBuffer();
        for (int row = HEIGHT - 1; row >= 0; row--) {
            for (int columnBit = row; columnBit < BIT_SIZE; columnBit += BITS_PER_COLUMN) {
                long mask = 1L << columnBit;
                if ((this.positions[FIRST_PLAYER] & mask) != 0)
                    bufferedString.append("X ");
                else if ((this.positions[SECOND_PLAYER] & mask) != 0)
                    bufferedString.append("O ");
                else 
                    bufferedString.append(". "); 
            }
            bufferedString.append("\n");
        }
        return bufferedString.toString();
    }
 
    @Override
    public boolean equals(Object otherObject) {
      
      if (otherObject == null || !(otherObject instanceof BinaryPlayground) ) {
        
        return false;
      }
      
      BinaryPlayground otherPlayground = (BinaryPlayground) otherObject;
      
      return Arrays.equals(this.positions, otherPlayground.positions);
    }
    
    
    public int hashCode() {
        long longHash = this.positions[FIRST_PLAYER] ^ this.positions[SECOND_PLAYER] * 31;
        return (int)(longHash ^ (longHash >>> 32));
    }
    
    
    //--------- helperMethods -----------------------------------------------------
    
    
    protected long[] makePosition(int[] arrayPosition) {
    	
        long[] position = new long[2];
        
        for (int index = 0; index < arrayPosition.length; index++){
            	
        	if (arrayPosition[index] != PlaygroundConstants.GREY){
        		
        		int row = (index - PlaygroundConstants.COLUMN_COUNT) / PlaygroundConstants.COLUMN_COUNT;
        		int column = (index - 1) % PlaygroundConstants.COLUMN_COUNT;
        		
                if (arrayPosition[index] == PlaygroundConstants.YELLOW)
                    position[FIRST_PLAYER] |= 1L << (column * 7 + row);
                
                else if (arrayPosition[index] == PlaygroundConstants.RED)
                    position[SECOND_PLAYER] |= 1L << (column * 7 + row);
            
        	}
        }
        	
        return position;
    }
    
    
    /**
     * @param positionArray
     * @return
     */
    protected int[] makeHeightOfColumn(ArrayPlayground positionArray) {
    	
        int[] heightOfColumns = positionArray.getColumnHeights();
        
        int[] result = new int[7];
        
        System.arraycopy(heightOfColumns, 0, result, 0, heightOfColumns.length);
        
        return result;
    }
    
    
    protected int[] makeFirstFreeBitOfColumn(int[] heightOfColumn) {
    	
        int[] firstFreeBitOfColumn = new int[7];
        
        for (int column = 0; column < WIDTH; column++)
            firstFreeBitOfColumn[column] = column * BITS_PER_COLUMN + heightOfColumn[column];
        
        return firstFreeBitOfColumn;
    }
}
