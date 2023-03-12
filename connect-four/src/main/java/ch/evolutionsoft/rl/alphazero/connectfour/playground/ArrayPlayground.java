package ch.evolutionsoft.rl.alphazero.connectfour.playground;

import java.util.ArrayList;
import java.util.List;

/**
 * @author evolutionsoft
 */
public class ArrayPlayground implements Playground {

	protected int[] playground; 
	
	protected int[] columnHeights;
	
	protected int fieldsLeft;
	
	public ArrayPlayground(){
		
		this.playground = new int[72];
		this.columnHeights = new int[7];
		
		this.fieldsLeft = 42;
		this.initializePlaygroundColors();
	}
	
	
	public ArrayPlayground(int[] position, int[] columnHeights){
		
		this.playground = position;
		this.columnHeights = columnHeights;
	}
	
	
	@Override	
	public int[] getPosition(){
		
		return this.playground;
	}
	
	
	public int[] getColumnHeights(){
		
		return this.columnHeights;
	}


	public int getFieldsLeft(){
		
		return this.fieldsLeft;
	}
	
	
	@Override
	public List<Integer> getAvailableColumns() {
		
		List<Integer> result = new ArrayList<>(7);
		
		for (int index = 0; index < 7; index++){
			
			if (this.columnHeights[PlaygroundConstants.columnsPrioritySorted[index]] < 6){
				
				result.add(PlaygroundConstants.columnsPrioritySorted[index]);
			}
		}
		
		return result;
	}
	
	
	@Override
	public boolean fourInARow(int lastColumn, int color) {

	  if (0 > lastColumn) {
	    
	    return false;
	  }
	  
		int lastPosition = this.getLastPosition(lastColumn);
		
		return this.fourInARowDiagonallyDown(lastPosition, color) ||
		       this.fourInARowDiagonallyUp(lastPosition, color) ||
		       this.fourInARowHorizontally(lastPosition, color) ||
		       this.fourInARowVertically(lastPosition, color);
	}

	
	@Override
	public int setField(int column, int color) {
		
        if (this.checkColumn(column)){
        	
        	if (!this.checkRow(this.columnHeights[column])){
        		
        		throw new IllegalArgumentException("Column already full.");
        	}
        	
        	int position = this.getPosition(column);
        	this.playground[position] = color;

        	int playedRow = this.columnHeights[column];
        	this.columnHeights[column]++;
        	
        	this.fieldsLeft--;
        	
        	return playedRow;
        }
        
        return -1;
	}

	
	@Override
	public int trySetField(int column, int color) {

    	int position = this.getPosition(column);
    	this.playground[position] = color;

      int playedRow = this.columnHeights[column];
    	this.columnHeights[column]++;
    	
    	this.fieldsLeft--;
    	
    	return playedRow;
	}

	
	@Override
	public int setFieldEmpty(int column) {
		
		if (this.checkColumn(column)){

          int emptiedRow = this.columnHeights[column];
			this.columnHeights[column]--;
			
			this.fieldsLeft++;
			
			if (!this.checkRow(this.columnHeights[column])){
				
				throw new IllegalArgumentException("Column already empty.");
			}
			
			int position = this.getPosition(column);
			this.playground[position] = PlaygroundConstants.EMPTY;
			
			return emptiedRow;
		}
		return -1;
	}

	
	@Override
	public int trySetFieldEmpty(int column) {

      int emptiedRow = this.columnHeights[column];
		this.columnHeights[column]--;
		
		this.fieldsLeft++;
		
		int position = this.getPosition(column);
		this.playground[position] = PlaygroundConstants.EMPTY;
		
		return emptiedRow;
	}


	public void reset(){
		
		this.playground = new int[72];
		this.columnHeights = new int[7];
		
		this.fieldsLeft = 42;
		this.initializePlaygroundColors();
	}
	
	
	public boolean isValidMove(int column){
		
		if (!this.checkColumn(column)){
			
			throw new IllegalArgumentException("Column " + column + " out of range.");
		}
		
		return this.columnHeights[column] < 6;
	}
	
	
	public int getFreeRow(int column){
	
		if (!this.checkColumn(column)){
			
			throw new IllegalArgumentException("Column " + column + " out of range.");
		}
		
		return this.columnHeights[column];
	}
	
	
	public int getRowFromIndex(int index){
		
		if (index < 0 || index >= 72 || this.playground[index] == PlaygroundConstants.GREY){
			
			throw new IllegalArgumentException("invalid index for row and column calculation.");
		}
		
		return (index - PlaygroundConstants.ARRAY_COLUMN_COUNT) / PlaygroundConstants.ARRAY_COLUMN_COUNT;
	}
	
	
	public int getColumnFromIndex(int index){
		
		if (index < 0 || index >= 72 || this.playground[index] == PlaygroundConstants.GREY){
			
			throw new IllegalArgumentException("invalid index for row and column calculation.");
		}
		
		return (index - 1) % PlaygroundConstants.ARRAY_COLUMN_COUNT;
	}


	public Line getWinningRow(int lastColumn, int color) {
		
		int position = this.getLastPosition(lastColumn);
		int forwardCount = 0;
		int backwardCount = 0;
		int direction = 0;
		
		if (this.fourInARowVertically(position, color)){

			direction = PlaygroundConstants.ARRAY_COLUMN_COUNT;
		}
		else if (this.fourInARowHorizontally(position, color)){

			direction = 1;
		}
		else if (this.fourInARowDiagonallyDown(position, color)){

			direction = PlaygroundConstants.ARRAY_COLUMN_COUNT - 1;
		}
		else if (this.fourInARowDiagonallyUp(position, color)){
			
			direction = PlaygroundConstants.ARRAY_COLUMN_COUNT + 1;
		}

		forwardCount = this.countStonesForward(position + direction, color, direction);
		backwardCount = this.countStonesBackward(position - direction, color, direction);
		
		if (direction > 0 && forwardCount + backwardCount >= 3){
			
			Field beginningField = new Field(position + direction * forwardCount);
			Field endField = new Field(position - direction * backwardCount);
				
			return new Line(beginningField, endField, color);
		}
		
		return null;
	}
		
	
	protected int countStonesForward(int position, int color, int direction){
		
		if (this.playground[position] != color){
			
			return 0;
		}
		
		return 1 + countStonesForward(position + direction, color, direction);
	}
	
	
	protected int countStonesBackward(int position, int color, int direction){
		
		if (this.playground[position] != color){
			
			return 0;
		}

		return 1 + countStonesBackward(position - direction, color, direction);
	}

		
	protected boolean fourInARowHorizontally(int lastPosition, int color){
		
		int stoneCount = 0;
		
		for (int position = lastPosition; 
		     stoneCount < 4 && this.playground[position] != PlaygroundConstants.GREY;
		     position++){
			
			if (this.playground[position] == color){
				
				stoneCount++;
			}
			else{
				break;
			}
		}		
		
		for (int position = lastPosition - 1; 
	         stoneCount < 4 && this.playground[position] != PlaygroundConstants.GREY;
	         position--){
		
			if (this.playground[position] == color){
			
				stoneCount++;
			}
			else{
				break;
			}
		}
		
		return stoneCount >= 4;
	}
	
	
	protected boolean fourInARowDiagonallyUp(int lastPosition, int color){
		
		int stoneCount = 0;
		
		for (int position = lastPosition; 
		     stoneCount < 4 && this.playground[position] != PlaygroundConstants.GREY;
		     position += PlaygroundConstants.ARRAY_COLUMN_COUNT + 1){
			
			if (this.playground[position] == color){
				
				stoneCount++;
			}
			else{
				break;
			}
		}		
		
		for (int position = lastPosition - PlaygroundConstants.ARRAY_COLUMN_COUNT - 1; 
	         stoneCount < 4 && this.playground[position] != PlaygroundConstants.GREY;
	         position -= PlaygroundConstants.ARRAY_COLUMN_COUNT + 1){
		
			if (this.playground[position] == color){
			
				stoneCount++;
			}
			else{
				break;
			}
		}
		
		return stoneCount >= 4;
	}

	
	protected boolean fourInARowDiagonallyDown(int lastPosition, int color){
		
		int stoneCount = 0;
		
		for (int position = lastPosition; 
		     stoneCount < 4 && this.playground[position] != PlaygroundConstants.GREY;
		     position += PlaygroundConstants.ARRAY_COLUMN_COUNT - 1){
			
			if (this.playground[position] == color){
				
				stoneCount++;
			}
			else{
				break;
			}
		}		
		
		for (int position = lastPosition - PlaygroundConstants.ARRAY_COLUMN_COUNT + 1; 
	         stoneCount < 4 && this.playground[position] != PlaygroundConstants.GREY;
	         position -= PlaygroundConstants.ARRAY_COLUMN_COUNT - 1){
		
			if (this.playground[position] == color){
			
				stoneCount++;
			}
			else{
				break;
			}
		}
		
		return stoneCount >= 4;
	}

	
	protected boolean fourInARowVertically(int lastPosition, int color){	
	
		int stoneCount = 0;
		
		for (int position = lastPosition; 
		     stoneCount < 4 && this.playground[position] != PlaygroundConstants.GREY;
		     position += PlaygroundConstants.ARRAY_COLUMN_COUNT){
			
			if (this.playground[position] == color){
				
				stoneCount++;
			}
			else{
				break;
			}
		}		
		
		for (int position = lastPosition - PlaygroundConstants.ARRAY_COLUMN_COUNT; 
	         stoneCount < 4 && this.playground[position] != PlaygroundConstants.GREY;
	         position -= PlaygroundConstants.ARRAY_COLUMN_COUNT){
		
			if (this.playground[position] == color){
			
				stoneCount++;
			}
			else{
				break;
			}
		}
		
		return stoneCount >= 4;
	}
	
	
	protected int getPosition(int column){
		
		return (this.columnHeights[column] + 1) * PlaygroundConstants.ARRAY_COLUMN_COUNT + column + 1;
	}
	
	
	protected int getLastPosition(int column){
		
		return this.columnHeights[column] * PlaygroundConstants.ARRAY_COLUMN_COUNT + column + 1;
	}
	
	
	protected boolean checkColumn(int column){
		
		return column >= 0 && column < 7;
	}
	
	
	protected boolean checkRow(int row){
		
		return row >= 0 && row < 6;
	}
	
	
	protected boolean checkColor(int color){
		
		return color == PlaygroundConstants.YELLOW || color == PlaygroundConstants.RED;
	}
	
	
	protected int otherColor(int color){
		
		if (color == PlaygroundConstants.YELLOW){
			
			return PlaygroundConstants.RED;
		}
		
		return PlaygroundConstants.YELLOW;
	}
	
	
	private void initializePlaygroundColors(){

		for (int index = 0; index < this.playground.length; index ++){
			
			this.playground[index] = PlaygroundConstants.EMPTY;
		} 
		
		for (int position = PlaygroundConstants.UPPER_LEFT; position <= PlaygroundConstants.UPPER_RIGHT; position++){
			
			this.playground[position] = PlaygroundConstants.GREY;
		}
        for (int position = PlaygroundConstants.LOWER_LEFT; position <= PlaygroundConstants.LOWER_RIGHT; position++){
			
			this.playground[position] = PlaygroundConstants.GREY;
		}
        
        for (int position = PlaygroundConstants.UPPER_LEFT; position <= PlaygroundConstants.LOWER_LEFT; position += PlaygroundConstants.ARRAY_COLUMN_COUNT){
			
			this.playground[position] = PlaygroundConstants.GREY;
		}     
        for (int position = PlaygroundConstants.UPPER_RIGHT; position <= PlaygroundConstants.LOWER_RIGHT; position += PlaygroundConstants.ARRAY_COLUMN_COUNT){
			
			this.playground[position] = PlaygroundConstants.GREY;
		}
	}
	
	
	public String toString(){
		
		StringBuilder result = new StringBuilder();
		
		for (int row = PlaygroundConstants.ARRAY_ROW_COUNT - 1; row >= 0; row--) {
			
			for (int column = 0; column < PlaygroundConstants.ARRAY_COLUMN_COUNT; column++) {
				
				int fieldColor = this.playground[row * PlaygroundConstants.ARRAY_COLUMN_COUNT + column];
				
				result = addFieldColor(result, fieldColor);
			}
			result.append(System.lineSeparator());
		}
		
		return result.toString();
	}


  StringBuilder addFieldColor(StringBuilder currentPlayground, int fieldColor) {

    switch(fieldColor){
    	
    	case PlaygroundConstants.EMPTY : {
    		
    		return appendFieldColorEmpty(currentPlayground);
    	}
    	case PlaygroundConstants.YELLOW : {
    		
    		return appendFieldColorYellow(currentPlayground);
    	}
    	case PlaygroundConstants.RED : {
    		
    		return appendFieldColorRed(currentPlayground);
    	}
    	default:{
    		
    		return appendFieldColorGrey(currentPlayground);
    	}
    }
  }


  StringBuilder appendFieldColorEmpty(StringBuilder currentPlayground) {
   
    return currentPlayground.append(". ");
  }

  StringBuilder appendFieldColorYellow(StringBuilder currentPlayground) {
   
    return currentPlayground.append("X ");
  }

  StringBuilder appendFieldColorRed(StringBuilder currentPlayground) {
   
    return currentPlayground.append("O ");
  }

  StringBuilder appendFieldColorGrey(StringBuilder currentPlayground) {
   
    return currentPlayground.append("* ");
  }
}
