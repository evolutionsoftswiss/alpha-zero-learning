package ch.evolutionsoft.rl.alphazero.connectfour.model;

import java.util.List;

/**
 * @author evolutionsoft
 */
public interface Playground {
	
	public Object getPosition();
	
	public int setField(int column, int color);
	
	public int setFieldEmpty(int column, int color);
	
	public int trySetField(int column, int color);

	public int trySetFieldEmpty(int column, int color);
	
	public boolean isValidMove(int column);
	
	public boolean fourInARow(int lastMove, int color);
	
	public List<Integer> getAvailableColumns();
}
