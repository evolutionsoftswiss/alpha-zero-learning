package ch.evolutionsoft.rl.alphazero.connectfour.model;

import static ch.evolutionsoft.rl.alphazero.connectfour.model.ModelViewConstants.*;

import java.beans.PropertyChangeListener;
import java.beans.PropertyChangeSupport;
import java.util.ArrayDeque;
import java.util.Deque;

public class ConnectFourGame {

  private BinaryPlayground playground;
  private int turn = PlaygroundConstants.YELLOW;
  private AbstractPlayer yellowPlayer;
  private AbstractPlayer redPlayer;
  private AbstractPlayer winner = null;
  private Deque<Move> moves = new ArrayDeque<>();
  private Deque<Move> movesTookBack = new ArrayDeque<>();
  private Line winningRow;
  private PropertyChangeSupport propertyChangeSupport;

  public ConnectFourGame(AbstractPlayer player1, AbstractPlayer player2) {

    this.playground = new BinaryPlayground();

    this.propertyChangeSupport = new PropertyChangeSupport(this);
    this.setPlayers(player1, player2);
    this.setGameInstanceFromComputerPlayer();
  }

  // -------- getters
  // -------------------------------------------------------------

  public int getTurn() {
    return this.turn;
  }

  public int getOtherColor() {
    return (this.turn == PlaygroundConstants.YELLOW) ? PlaygroundConstants.RED : PlaygroundConstants.YELLOW;
  }

  public AbstractPlayer getCurrentPlayer() {
    return (this.turn == PlaygroundConstants.YELLOW) ? yellowPlayer : redPlayer;
  }

  public AbstractPlayer getOtherPlayer() {
    return (this.turn == PlaygroundConstants.YELLOW) ? redPlayer : yellowPlayer;
  }

  public AlphaZeroPlayer getComputerPlayer() {
    if (this.yellowPlayer instanceof AlphaZeroPlayer)
      return (AlphaZeroPlayer) this.yellowPlayer;
    if (this.redPlayer instanceof AlphaZeroPlayer)
      return (AlphaZeroPlayer) this.redPlayer;
    return null;
  }

  public int getFieldsLeft() {

    return this.playground.getFieldsLeft();
  }

  public AbstractPlayer getWinner() {

    return this.winner;
  }

  public BinaryPlayground getPlayGround() {

    return this.playground;
  }

  public String getGameState() {

    if (this.notOver()) {
      if (this.getCurrentPlayer() instanceof HumanPlayer) {
  
        return ((this.turn == PlaygroundConstants.YELLOW) ? "Yellow to move" : "Red to move");
      }
      if (!this.getComputerPlayer().searchInitialized() &&
          (this.yellowPlayer instanceof AlphaZeroPlayer || this.redPlayer instanceof AlphaZeroPlayer) ) {
  
        return ((this.turn == PlaygroundConstants.YELLOW) ? "Yellow (alpha zero) to move"
            : "Red (alpha zero) to move");
      }
      return "Computer searches move...";
    }

    if (this.winner == null) {
      return ("Game over: Draw");
    }

    return ((this.getWinner() == yellowPlayer) ? "Game over: Yellow wins " : "Game over: Red wins");
  }

  public Deque<Move> getMoveHistory() {

    return this.moves;
  }

  public Move getLastMove() {

    if (!moves.isEmpty()) {
      return this.moves.peek();
    }
    return null;
  }

  public Move getLastTookBackMove() {

    if (!this.movesTookBack.isEmpty()) {

      return this.movesTookBack.peek();
    }
    return null;
  }

  public Line getWinningRow() {
    return this.winningRow;
  }

  // --------- setters
  // ------------------------------------------------------------

  public void setPlayers(AbstractPlayer player1, AbstractPlayer player2) {

    if (player1 == null || player2 == null || player1 == player2
        || !haveLegalColors(player1, player2)) {

      throw new IllegalArgumentException("Illegal players or colors.");
    }
    if (player1.getColor() == PlaygroundConstants.YELLOW) {

      this.yellowPlayer = player1;
      this.redPlayer = player2;
    } else {

      this.yellowPlayer = player2;
      this.redPlayer = player1;
    }

    this.propertyChangeSupport.firePropertyChange(PLAYERS_PROPERTY, false, true);
  }

  public void setOtherPlayer(AbstractPlayer newPlayer) {

    if (newPlayer == null || newPlayer == this.getCurrentPlayer()
        || !this.haveLegalColors(newPlayer, this.getCurrentPlayer())) {

      throw new IllegalArgumentException("Illegal player or color");
    }
    
    if (this.turn == PlaygroundConstants.YELLOW) {

      redPlayer = newPlayer;

    } else if (this.turn == PlaygroundConstants.RED) {

      yellowPlayer = newPlayer;
    }

    this.setGameInstanceFromComputerPlayer();

    this.propertyChangeSupport.firePropertyChange(OTHER_PLAYER_PROPERTY, false, true);
  }

  // --------- public methods
  // ------------------------------------------------------------

  public void addPropertyChangeListener(PropertyChangeListener view) {

    this.propertyChangeSupport.addPropertyChangeListener(view);
  }

  public void newGame(AbstractPlayer player1, AbstractPlayer player2) {

    this.reset();
    this.setPlayers(player1, player2);
    this.setGameInstanceFromComputerPlayer();

    this.propertyChangeSupport.firePropertyChange(NEW_GAME_PROPERTY, false, true);
  }

  public void move(int column, int color) {

    if (this.notOver() && color == this.turn && this.playground.isValidMove(column)) {

      int playedRow = this.playground.setField(column, color);

      Move move = new Move(playedRow * PlaygroundConstants.COLUMN_COUNT + column, color);
      this.moves.push(move);
      if (!movesTookBack.isEmpty()) {
        movesTookBack.clear();
      }

      this.swapTurn();

      this.checkForWinner(column, color);

      if (this.getCurrentPlayer() instanceof AlphaZeroPlayer
          && this.notOver())
        ((AlphaZeroPlayer) this.getCurrentPlayer()).move();

      this.propertyChangeSupport.firePropertyChange(NEW_MOVE_PROPERTY, false, true);
    }
  }

  public void takeBackMove() {

    if (!this.moves.isEmpty()) {

      Move lastMove = moves.pop();

      this.movesTookBack.push(lastMove);

      this.playground.setFieldEmpty(lastMove.getColumn(), lastMove.getColor());

      this.swapTurn();

      this.propertyChangeSupport.firePropertyChange(MOVE_TOOK_BACK_PROPERTY, false, true);
    }
  }

  public void reDoMove() {

    if (!this.movesTookBack.isEmpty()) {

      Move move = this.movesTookBack.pop();
      this.moves.push(move);

      this.swapTurn();
      this.playground.setField(move.getColumn(), move.getColor());
      
      this.checkForWinner(move.getColumn(), move.getColor());

      this.propertyChangeSupport.firePropertyChange(MOVE_REDONE_PROPERTY, false, true);
    }
  }

  public void resetWinner() {

    this.winner = null;
    this.winningRow = null;
    this.propertyChangeSupport.firePropertyChange(RESET_WINNER_PROPERTY, false, true);
  }

  public boolean hasComputerPlayer() {
    return this.getComputerPlayer() != null;
  }

  public boolean notOver() {
    return (this.playground.getFieldsLeft() > 0 && this.winner == null);
  }

  // --------- helper methods
  // ------------------------------------------------------------

  /**
   * @param player1
   * @param player2
   * @return
   */
  protected boolean haveLegalColors(AbstractPlayer player1, AbstractPlayer player2) {
    return player1.getColor() == PlaygroundConstants.RED && player2.getColor() == PlaygroundConstants.YELLOW
        || player1.getColor() == PlaygroundConstants.YELLOW && player2.getColor() == PlaygroundConstants.RED;
  }

  protected void setGameInstanceFromComputerPlayer() {

    if (this.yellowPlayer instanceof AlphaZeroPlayer) {
      ((AlphaZeroPlayer) this.yellowPlayer).setGame(this);
    }

    if (this.redPlayer instanceof AlphaZeroPlayer) {
      ((AlphaZeroPlayer) this.redPlayer).setGame(this);
    }
  }

  protected void reset() {

    resetWinner();
    this.moves.clear();
    this.movesTookBack.clear();
    this.turn = PlaygroundConstants.YELLOW;
    this.playground.reset();

  }

  protected void checkForWinner(int column, int color) {

    if (this.playground.fourInARow(column, color)) {

      this.setWinner(color);
      this.winningRow = this.playground.getWinningRow(column, color);

      this.propertyChangeSupport.firePropertyChange(FOUR_IN_A_ROW_PROPERTY, false, true);
    }

  }

  protected void swapTurn() {

    this.turn = (this.turn == PlaygroundConstants.YELLOW) ? PlaygroundConstants.RED : PlaygroundConstants.YELLOW;
  }

  protected void setWinner(int color) {

    this.winner = (color == PlaygroundConstants.YELLOW) ? yellowPlayer : redPlayer;
  }
}
