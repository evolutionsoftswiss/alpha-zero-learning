package ch.evolutionsoft.rl.alphazero.connectfour.playground;

import java.util.ArrayList;
import java.util.List;
import java.util.Observable;

public class ArrayPlayground extends Observable implements Playground {

	protected int[] playground_; 
	
	protected int[] columnHeights_;
	
	protected int fieldsLeft_;
	
	public ArrayPlayground(){
		
		this.playground_ = new int[72];
		this.columnHeights_ = new int[7];
		
		this.fieldsLeft_ = 42;
		this.initializePlaygroundColors();
	}
	
	
	public ArrayPlayground(int[] position, int[] columnHeights){
		
		this.playground_ = position;
		this.columnHeights_ = columnHeights;
	}
	
	
	@Override	
	public int[] getPosition(){
		
		return this.playground_;
	}
	
	
	public int[] getColumnHeights(){
		
		return this.columnHeights_;
	}


	public int getFieldsLeft(){
		
		return this.fieldsLeft_;
	}
	
	
	@Override
	public List<Integer> getAvailableColumns() {
		
		List<Integer> result = new ArrayList<Integer>(7);
		
		for (int index = 0; index < 7; index++){
			
			if (this.columnHeights_[Playground.columnsPrioritySorted[index]] < 6){
				
				result.add(Playground.columnsPrioritySorted[index]);
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
        	
        	if (!this.checkRow(this.columnHeights_[column])){
        		
        		throw new IllegalArgumentException("Column already full.");
        	}
        	
        	int position = this.getPosition(column);
        	this.playground_[position] = color;

        	this.columnHeights_[column]++;
        	
        	this.fieldsLeft_--;
        	
        	this.setChanged();
        	this.notifyObservers("New Move");
        }
	}

	
	@Override
	public void trySetField(int column, int color) {

    	int position = this.getPosition(column);
    	this.playground_[position] = color;

    	this.columnHeights_[column]++;
    	
    	this.fieldsLeft_--;
	}

	
	@Override
	public void setFieldEmpty(int column) {
		
		if (this.checkColumn(column)){
			
			this.columnHeights_[column]--;
			
			this.fieldsLeft_++;
			
			if (!this.checkRow(this.columnHeights_[column])){
				
				throw new IllegalArgumentException("Column already empty.");
			}
			
			int position = this.getPosition(column);
			this.playground_[position] = Playground.EMPTY;
			
        	this.setChanged();
        	this.notifyObservers("Move took back");
		}
		
	}

	
	@Override
	public void trySetFieldEmpty(int column) {

		this.columnHeights_[column]--;
		
		this.fieldsLeft_++;
		
		int position = this.getPosition(column);
		this.playground_[position] = Playground.EMPTY;
	}


	public void reset(){
		
		this.playground_ = new int[72];
		this.columnHeights_ = new int[7];
		
		this.fieldsLeft_ = 42;
		this.initializePlaygroundColors();
		
		this.setChanged();
		this.notifyObservers("Reset");
	}
	
	
	public boolean isValidMove(int column){
		
		if (!this.checkColumn(column)){
			
			throw new IllegalArgumentException("Column " + column + " out of range.");
		}
		
		return this.columnHeights_[column] < 6;
	}
	
	
	public int getFreeRow(int column){
	
		if (!this.checkColumn(column)){
			
			throw new IllegalArgumentException("Column " + column + " out of range.");
		}
		
		return this.columnHeights_[column];
	}
	
	
	public int getRowFromIndex(int index){
		
		if (index < 0 || index >= 72 || this.playground_[index] == Playground.GREY){
			
			throw new IllegalArgumentException("invalid index for row and column calculation.");
		}
		
		return index - ArrayPlaygroundConstants.COLUMN_COUNT / ArrayPlaygroundConstants.COLUMN_COUNT;
	}
	
	
	public int getColumnFromIndex(int index){
		
		if (index < 0 || index >= 72 || this.playground_[index] == Playground.GREY){
			
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
		
		if (this.playground_[position] != color){
			
			return 0;
		}
		
		return 1 + countStonesForward(position + direction, color, direction);
	}
	
	
	protected int countStonesBackward(int position, int color, int direction){
		
		if (this.playground_[position] != color){
			
			return 0;
		}

		return 1 + countStonesBackward(position - direction, color, direction);
	}

		
	protected boolean fourInARowHorizontally(int lastPosition, int color){
		
		int stoneCount = 0;
		
		for (int position = lastPosition; 
		     stoneCount < 4 && this.playground_[position] != Playground.GREY;
		     position++){
			
			if (this.playground_[position] == color){
				
				stoneCount++;
			}
			else{
				break;
			}
		}		
		
		for (int position = lastPosition - 1; 
	         stoneCount < 4 && this.playground_[position] != Playground.GREY;
	         position--){
		
			if (this.playground_[position] == color){
			
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
		     stoneCount < 4 && this.playground_[position] != Playground.GREY;
		     position += ArrayPlaygroundConstants.COLUMN_COUNT + 1){
			
			if (this.playground_[position] == color){
				
				stoneCount++;
			}
			else{
				break;
			}
		}		
		
		for (int position = lastPosition - ArrayPlaygroundConstants.COLUMN_COUNT - 1; 
	         stoneCount < 4 && this.playground_[position] != Playground.GREY;
	         position -= ArrayPlaygroundConstants.COLUMN_COUNT + 1){
		
			if (this.playground_[position] == color){
			
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
		     stoneCount < 4 && this.playground_[position] != Playground.GREY;
		     position += ArrayPlaygroundConstants.COLUMN_COUNT - 1){
			
			if (this.playground_[position] == color){
				
				stoneCount++;
			}
			else{
				break;
			}
		}		
		
		for (int position = lastPosition - ArrayPlaygroundConstants.COLUMN_COUNT + 1; 
	         stoneCount < 4 && this.playground_[position] != Playground.GREY;
	         position -= ArrayPlaygroundConstants.COLUMN_COUNT - 1){
		
			if (this.playground_[position] == color){
			
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
		     stoneCount < 4 && this.playground_[position] != Playground.GREY;
		     position += ArrayPlaygroundConstants.COLUMN_COUNT){
			
			if (this.playground_[position] == color){
				
				stoneCount++;
			}
			else{
				break;
			}
		}		
		
		for (int position = lastPosition - ArrayPlaygroundConstants.COLUMN_COUNT; 
	         stoneCount < 4 && this.playground_[position] != Playground.GREY;
	         position -= ArrayPlaygroundConstants.COLUMN_COUNT){
		
			if (this.playground_[position] == color){
			
				stoneCount++;
			}
			else{
				break;
			}
		}
		
		return stoneCount >= 4;
	}
	
	
	protected int getPosition(int column){
		
		return (this.columnHeights_[column] + 1) * ArrayPlaygroundConstants.COLUMN_COUNT + column + 1;
	}
	
	
	protected int getLastPosition(int column){
		
		return this.columnHeights_[column] * ArrayPlaygroundConstants.COLUMN_COUNT + column + 1;
	}
	
	
	protected boolean checkColumn(int column){
		
		return column >= 0 && column < 7;
	}
	
	
	protected boolean checkRow(int row){
		
		return row >= 0 && row < 6;
	}
	
	
	protected boolean checkColor(int color){
		
		return color == Playground.YELLOW || color == Playground.RED;
	}
	
	
	protected int otherColor(int color){
		
		if (color == Playground.YELLOW){
			
			return Playground.RED;
		}
		
		return Playground.YELLOW;
	}
	
	
	private void initializePlaygroundColors(){

		for (int index = 0; index < this.playground_.length; index ++){
			
			this.playground_[index] = Playground.EMPTY;
		} 
		
		for (int position = ArrayPlaygroundConstants.UPPER_LEFT; position <= ArrayPlaygroundConstants.UPPER_RIGHT; position++){
			
			this.playground_[position] = Playground.GREY;
		}
        for (int position = ArrayPlaygroundConstants.LOWER_LEFT; position <= ArrayPlaygroundConstants.LOWER_RIGHT; position++){
			
			this.playground_[position] = Playground.GREY;
		}
        
        for (int position = ArrayPlaygroundConstants.UPPER_LEFT; position <= ArrayPlaygroundConstants.LOWER_LEFT; position += ArrayPlaygroundConstants.COLUMN_COUNT){
			
			this.playground_[position] = Playground.GREY;
		}     
        for (int position = ArrayPlaygroundConstants.UPPER_RIGHT; position <= ArrayPlaygroundConstants.LOWER_RIGHT; position += ArrayPlaygroundConstants.COLUMN_COUNT){
			
			this.playground_[position] = Playground.GREY;
		}
	}
	
	
	public String toString(){
		
		String result = "";
		
		for (int row = ArrayPlaygroundConstants.ROW_COUNT - 1; row >= 0; row--){
			
			for (int column = 0; column < ArrayPlaygroundConstants.COLUMN_COUNT; column++){
				
				int fieldColor = this.playground_[row * ArrayPlaygroundConstants.COLUMN_COUNT + column];
				
				switch(fieldColor){
					
					case Playground.EMPTY : {
						
						result += ". ";
						break;
					}
					case Playground.YELLOW : {
						
						result += "X ";
						break;
					}
					case Playground.RED : {
						
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
