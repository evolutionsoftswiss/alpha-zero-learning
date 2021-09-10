package ch.evolutionsoft.rl.alphazero.connectfour.playground;

import java.util.List;

public interface Playground {

	public final static int EMPTY = 2;
	public final static int YELLOW = 0;
	public final static int RED = 1;
	public static int GREY = 3;
	
	public final static int[] columnsPrioritySorted = new int[]{3, 2, 4, 1, 5, 0, 6};
	
	public Object getPosition();
	
	public void setField(int column, int color);
	
	public void setFieldEmpty(int column);
	
	public void trySetField(int column, int color);

	public void trySetFieldEmpty(int column);
	
	public boolean isValidMove(int column);
	
	public boolean fourInARow(int lastMove, int color);
	
	public List<Integer> getAvailableColumns();
}
