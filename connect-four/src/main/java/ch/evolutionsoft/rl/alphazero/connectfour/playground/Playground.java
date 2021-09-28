package ch.evolutionsoft.rl.alphazero.connectfour.playground;

import java.util.List;

/**
 * @author evolutionsoft
 */
public interface Playground {
	
	public Object getPosition();
	
	public int setField(int column, int color);
	
	public int setFieldEmpty(int column);
	
	public int trySetField(int column, int color);

	public int trySetFieldEmpty(int column);
	
	public boolean isValidMove(int column);
	
	public boolean fourInARow(int lastMove, int color);
	
	public List<Integer> getAvailableColumns();
}
