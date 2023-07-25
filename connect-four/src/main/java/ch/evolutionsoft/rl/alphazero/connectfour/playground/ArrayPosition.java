package ch.evolutionsoft.rl.alphazero.connectfour.playground;

/**
 * @author evolutionsoft
 */
public class ArrayPosition {

	private int[] playground;
	
	private int[] columnHeights;
	
	private int fieldsLeft;
	
	public ArrayPosition(int[] position, int[] columnHeights){
		
		this.playground = position;
		this.columnHeights = columnHeights;
    
    this.fieldsLeft = 42;
    
    for (int row = PlaygroundConstants.ARRAY_ROW_COUNT - 1; row >= 0; row--) {
      
      for (int column = 0; column < PlaygroundConstants.ARRAY_COLUMN_COUNT; column++) {
        
        int index = row * PlaygroundConstants.ARRAY_COLUMN_COUNT + column;
        
        if (this.playground[index] == PlaygroundConstants.YELLOW ||
            this.playground[index] == PlaygroundConstants.RED) {
          
          this.fieldsLeft--;
        }
      }
    }
	}

	public int[] getPosition(){
		
	  int[] positionCopy = new int[this.playground.length];
		System.arraycopy(this.playground, 0, positionCopy, 0, this.playground.length);
	  
	  return positionCopy;
	}
	
	public int[] getColumnHeights(){

    int[] columnHeightsCopy = new int[this.columnHeights.length];
    System.arraycopy(this.columnHeights, 0, columnHeightsCopy, 0, this.columnHeights.length);
	  
		return columnHeightsCopy;
	}

	public int getFieldsLeft(){
		
		return this.fieldsLeft;
	}
	
}