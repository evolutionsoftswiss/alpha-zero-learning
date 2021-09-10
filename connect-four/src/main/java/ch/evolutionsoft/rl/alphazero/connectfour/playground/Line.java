/*
 * Created on 30.10.2004
 *
 * TODO To change the template for this generated file go to
 * Window - Preferences - Java - Code Style - Code Templates
 */
package ch.evolutionsoft.rl.alphazero.connectfour.playground;

/**
 * @author Markus Bloesch
 *
 * TODO To change the template for this generated type comment go to
 * Window - Preferences - Java - Code Style - Code Templates
 */
public class Line{
    
    protected Field beginning_, end_;
    protected int color_;
    
    public Line(Field beginning, Field end, int color) {
    	
    	this.beginning_ = beginning;
    	this.end_ = end;
    	this.color_ = color;
    }
    
    public Field getBeginning() {
    	
        return this.beginning_;   
    }
    
    public Field getEnd() {
    	
        return this.end_;   
    }
    
    public int getColor() {
    	
        return this.color_;   
    }
    
    public boolean equals(Object object) {
    	
        if (object.getClass() == this.getClass())
            return this.equals((Line)object);
        
        return false;
    }
    
    public boolean equals(Line otherThreat) {
    	
        return this.beginning_.equals(otherThreat.beginning_)
            && this.end_.equals(otherThreat.end_)
            && this.color_ == otherThreat.getColor();
    }
    
    public String toString() {
    	
        return "beginning: " + this.beginning_.toString() +
               "end: " + this.end_.toString() +
               this.color_;
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
