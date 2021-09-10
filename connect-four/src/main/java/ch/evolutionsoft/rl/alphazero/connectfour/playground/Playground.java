package ch.evolutionsoft.rl.alphazero.connectfour.playground;

import java.util.List;

/**
 * 
 * @author evolutionsoft
 *
 */
public interface Playground {
	
	public Object getPosition();
	
	public void setField(int column, int color);
	
	public void setFieldEmpty(int column);
	
	public void trySetField(int column, int color);

	public void trySetFieldEmpty(int column);
	
	public boolean isValidMove(int column);
	
	public boolean fourInARow(int lastMove, int color);
	
	public List<Integer> getAvailableColumns();
}
