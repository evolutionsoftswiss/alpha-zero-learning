package ch.evolutionsoft.rl.alphazero.connectfour.model;

public class HumanPlayer extends AbstractPlayer {
    
	/**
	 * @param color
	 */
	public HumanPlayer(int color) {
		super(color);
	}
    
	/* (non-Javadoc)
	 * @see model.AbstractPlayer#move(model.Game)
	 */
	public void move(ConnectFourGame game, int column) {
		game.move(column, this.getColor());
	}
}
