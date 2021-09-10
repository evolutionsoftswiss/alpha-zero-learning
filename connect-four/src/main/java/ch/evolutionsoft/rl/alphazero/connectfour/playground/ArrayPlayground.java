package ch.evolutionsoft.rl.alphazero.connectfour.playground;

import java.util.ArrayList;
import java.util.List;

/**
 * 
 * @author evolutionsoft
 *
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
		
		List<Integer> result = new ArrayList<Integer>(7);
		
		for (int index = 0; index < 7; index++){
			
			if (this.columnHeights[ArrayPlaygroundConstants.columnsPrioritySorted[index]] < 6){
				
				result.add(ArrayPlaygroundConstants.columnsPrioritySorted[index]);
			}
		}
		
		return result;
	}
	
	
	@Override
	public boolean fourInARow(int lastColumn, int color) {
		
		int lastPosition = this.getLastPosition(lastColumn);
		
		return this.fourInARowDiagonallyDown(lastPosition, color) ||
		       this.fourInARowDiagonallyUp(lastPosition, color) ||
		       this.fourInARowHorizontally(lastPosition, color) ||
		       this.fourInARowVertically(lastPosition, color);
	}

	
	@Override
	public void setField(int column, int color) {
		
        if (this.checkColumn(column)){
        	
        	if (!this.checkRow(this.columnHeights[column])){
        		
        		throw new IllegalArgumentException("Column already full.");
        	}
        	
        	int position = this.getPosition(column);
        	this.playground[position] = color;

        	this.columnHeights[column]++;
        	
        	this.fieldsLeft--;
        }
	}

	
	@Override
	public void trySetField(int column, int color) {

    	int position = this.getPosition(column);
    	this.playground[position] = color;

    	this.columnHeights[column]++;
    	
    	this.fieldsLeft--;
	}

	
	@Override
	public void setFieldEmpty(int column) {
		
		if (this.checkColumn(column)){
			
			this.columnHeights[column]--;
			
			this.fieldsLeft++;
			
			if (!this.checkRow(this.columnHeights[column])){
				
				throw new IllegalArgumentException("Column already empty.");
			}
			
			int position = this.getPosition(column);
			this.playground[position] = ArrayPlaygroundConstants.EMPTY;
		}
		
	}

	
	@Override
	public void trySetFieldEmpty(int column) {

		this.columnHeights[column]--;
		
		this.fieldsLeft++;
		
		int position = this.getPosition(column);
		this.playground[position] = ArrayPlaygroundConstants.EMPTY;
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
		
		if (index < 0 || index >= 72 || this.playground[index] == ArrayPlaygroundConstants.GREY){
			
			throw new IllegalArgumentException("invalid index for row and column calculation.");
		}
		
		return index - ArrayPlaygroundConstants.COLUMN_COUNT / ArrayPlaygroundConstants.COLUMN_COUNT;
	}
	
	
	public int getColumnFromIndex(int index){
		
		if (index < 0 || index >= 72 || this.playground[index] == ArrayPlaygroundConstants.GREY){
			
			throw new IllegalArgumentException("invalid index for row and column calculation.");
		}
		
		return (index - 1) % ArrayPlaygroundConstants.COLUMN_COUNT;
	}


	public Line getWinningRow(int lastColumn, int color) {
		
		int position = this.getLastPosition(lastColumn);
		int forwardCount = 0;
		int backwardCount = 0;
		int direction = 0;
		
		if (this.fourInARowVertically(position, color)){

			direction = ArrayPlaygroundConstants.COLUMN_COUNT;
		}
		else if (this.fourInARowHorizontally(position, color)){

			direction = 1;
		}
		else if (this.fourInARowDiagonallyDown(position, color)){

			direction = ArrayPlaygroundConstants.COLUMN_COUNT - 1;
		}
		else if (this.fourInARowDiagonallyUp(position, color)){
			
			direction = ArrayPlaygroundConstants.COLUMN_COUNT + 1;
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
		     stoneCount < 4 && this.playground[position] != ArrayPlaygroundConstants.GREY;
		     position++){
			
			if (this.playground[position] == color){
				
				stoneCount++;
			}
			else{
				break;
			}
		}		
		
		for (int position = lastPosition - 1; 
	         stoneCount < 4 && this.playground[position] != ArrayPlaygroundConstants.GREY;
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
		     stoneCount < 4 && this.playground[position] != ArrayPlaygroundConstants.GREY;
		     position += ArrayPlaygroundConstants.COLUMN_COUNT + 1){
			
			if (this.playground[position] == color){
				
				stoneCount++;
			}
			else{
				break;
			}
		}		
		
		for (int position = lastPosition - ArrayPlaygroundConstants.COLUMN_COUNT - 1; 
	         stoneCount < 4 && this.playground[position] != ArrayPlaygroundConstants.GREY;
	         position -= ArrayPlaygroundConstants.COLUMN_COUNT + 1){
		
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
		     stoneCount < 4 && this.playground[position] != ArrayPlaygroundConstants.GREY;
		     position += ArrayPlaygroundConstants.COLUMN_COUNT - 1){
			
			if (this.playground[position] == color){
				
				stoneCount++;
			}
			else{
				break;
			}
		}		
		
		for (int position = lastPosition - ArrayPlaygroundConstants.COLUMN_COUNT + 1; 
	         stoneCount < 4 && this.playground[position] != ArrayPlaygroundConstants.GREY;
	         position -= ArrayPlaygroundConstants.COLUMN_COUNT - 1){
		
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
		     stoneCount < 4 && this.playground[position] != ArrayPlaygroundConstants.GREY;
		     position += ArrayPlaygroundConstants.COLUMN_COUNT){
			
			if (this.playground[position] == color){
				
				stoneCount++;
			}
			else{
				break;
			}
		}		
		
		for (int position = lastPosition - ArrayPlaygroundConstants.COLUMN_COUNT; 
	         stoneCount < 4 && this.playground[position] != ArrayPlaygroundConstants.GREY;
	         position -= ArrayPlaygroundConstants.COLUMN_COUNT){
		
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
		
		return (this.columnHeights[column] + 1) * ArrayPlaygroundConstants.COLUMN_COUNT + column + 1;
	}
	
	
	protected int getLastPosition(int column){
		
		return this.columnHeights[column] * ArrayPlaygroundConstants.COLUMN_COUNT + column + 1;
	}
	
	
	protected boolean checkColumn(int column){
		
		return column >= 0 && column < 7;
	}
	
	
	protected boolean checkRow(int row){
		
		return row >= 0 && row < 6;
	}
	
	
	protected boolean checkColor(int color){
		
		return color == ArrayPlaygroundConstants.YELLOW || color == ArrayPlaygroundConstants.RED;
	}
	
	
	protected int otherColor(int color){
		
		if (color == ArrayPlaygroundConstants.YELLOW){
			
			return ArrayPlaygroundConstants.RED;
		}
		
		return ArrayPlaygroundConstants.YELLOW;
	}
	
	
	private void initializePlaygroundColors(){

		for (int index = 0; index < this.playground.length; index ++){
			
			this.playground[index] = ArrayPlaygroundConstants.EMPTY;
		} 
		
		for (int position = ArrayPlaygroundConstants.UPPER_LEFT; position <= ArrayPlaygroundConstants.UPPER_RIGHT; position++){
			
			this.playground[position] = ArrayPlaygroundConstants.GREY;
		}
        for (int position = ArrayPlaygroundConstants.LOWER_LEFT; position <= ArrayPlaygroundConstants.LOWER_RIGHT; position++){
			
			this.playground[position] = ArrayPlaygroundConstants.GREY;
		}
        
        for (int position = ArrayPlaygroundConstants.UPPER_LEFT; position <= ArrayPlaygroundConstants.LOWER_LEFT; position += ArrayPlaygroundConstants.COLUMN_COUNT){
			
			this.playground[position] = ArrayPlaygroundConstants.GREY;
		}     
        for (int position = ArrayPlaygroundConstants.UPPER_RIGHT; position <= ArrayPlaygroundConstants.LOWER_RIGHT; position += ArrayPlaygroundConstants.COLUMN_COUNT){
			
			this.playground[position] = ArrayPlaygroundConstants.GREY;
		}
	}
	
	
	public String toString(){
		
		String result = "";
		
		for (int row = ArrayPlaygroundConstants.ROW_COUNT - 1; row >= 0; row--){
			
			for (int column = 0; column < ArrayPlaygroundConstants.COLUMN_COUNT; column++){
				
				int fieldColor = this.playground[row * ArrayPlaygroundConstants.COLUMN_COUNT + column];
				
				switch(fieldColor){
					
					case ArrayPlaygroundConstants.EMPTY : {
						
						result += ". ";
						break;
					}
					case ArrayPlaygroundConstants.YELLOW : {
						
						result += "X ";
						break;
					}
					case ArrayPlaygroundConstants.RED : {
						
						result += "O ";
						break;
					}
					default:{
						
						result += "* ";
					}
				}
			}
			result += "\n";
		}
		
		return result;
	}
}
