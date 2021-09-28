package ch.evolutionsoft.rl.alphazero.connectfour.playground;

/**
 * @author evolutionsoft
 */
public class Line{
    
    protected Field beginning;
    protected Field end;
    protected int color;
    
    public Line(Field beginning, Field end, int color) {
    	
    	this.beginning = beginning;
    	this.end = end;
    	this.color = color;
    }
    
    public Field getBeginning() {
    	
        return this.beginning;   
    }
    
    public Field getEnd() {
    	
        return this.end;   
    }
    
    public int getColor() {
    	
        return this.color;   
    }
    
    public boolean equalsLine(Line otherThreat) {
    	
        return this.beginning.equalsField(otherThreat.beginning)
            && this.end.equalsField(otherThreat.end)
            && this.color == otherThreat.getColor();
    }
    
    public String toString() {
    	
        return "beginning: " + this.beginning.toString() +
               "end: " + this.end.toString() +
               this.color;
    }


	public int getWinningRowLength() {
		
        int winningRowLength = 
        	this.getEnd().getColumn() - this.getBeginning().getColumn();
        if (winningRowLength < 4)//in case of vertical winning row
            winningRowLength = 4;
        return winningRowLength;
	}

	
	public int getWinningRowColumnDirection() {
		
        int columnDifference = 
        	this.getEnd().getColumn() - this.getBeginning().getColumn();
        
        if (columnDifference < 0)
            return -1;
        else if (columnDifference > 0)
            return 1;
        else 
            return 0;
	}

	
	public int getWinningRowRowDirection() {
		
        int rowDifference = this.getEnd().getRow() - this.getBeginning().getRow();
        
        if (rowDifference < 0)
            return -1;
        else if (rowDifference > 0)
            return 1;
        else 
            return 0;
	}
}
