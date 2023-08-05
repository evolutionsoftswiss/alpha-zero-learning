package ch.evolutionsoft.rl.alphazero.connectfour.model;

public class GameDriver {

  public static final String DEFAULT_MODEL_NAME = "model.bin";
  public static final int TWO_PLAYERS = 2;
  public static final int ONE_PLAYER = 1;

  private ConnectFourGame game;
  private AbstractPlayer yellowPlayer;
  private AbstractPlayer redPlayer;
  private int currentNumberOfPlayers = ONE_PLAYER;
  private boolean useMonteCarloSearch = true;

  public GameDriver() {
    yellowPlayer = new HumanPlayer(PlaygroundConstants.YELLOW);
    redPlayer = new AlphaZeroPlayer(PlaygroundConstants.RED, DEFAULT_MODEL_NAME, useMonteCarloSearch, 0);
    game = new ConnectFourGame(yellowPlayer, redPlayer);
  }

  public ConnectFourGame getGame() {
    return this.game;
  }

  public void newGame() {

    if (this.currentNumberOfPlayers == TWO_PLAYERS) {
      this.yellowPlayer = new HumanPlayer(PlaygroundConstants.YELLOW);
      this.redPlayer = new HumanPlayer(PlaygroundConstants.RED);
    } else if (this.currentNumberOfPlayers == ONE_PLAYER) {
      this.yellowPlayer = new HumanPlayer(PlaygroundConstants.YELLOW);
      this.redPlayer = new AlphaZeroPlayer(PlaygroundConstants.RED, DEFAULT_MODEL_NAME, useMonteCarloSearch, 0);
    }
    this.game.newGame(yellowPlayer, redPlayer);
  }

  public void takeBackMove() {

    if (game.hasComputerPlayer() && !game.getComputerPlayer().searchInitialized()) {
      this.game.takeBackMove();
    }
  }

  public void reDoMove() {
 
    if (game.hasComputerPlayer() && !game.getComputerPlayer().searchInitialized()) {
      this.game.reDoMove();
    }
  }

  public void setUseMonteCarloSearch(boolean useMonteCarloSearchOption) {
    
    this.useMonteCarloSearch = useMonteCarloSearchOption;
    if (game.getCurrentPlayer() instanceof AlphaZeroPlayer) {
      ((AlphaZeroPlayer)game.getCurrentPlayer()).setUseMonteCarloSearch(useMonteCarloSearchOption);
    } else if (game.getOtherPlayer() instanceof AlphaZeroPlayer) {
      ((AlphaZeroPlayer)game.getOtherPlayer()).setUseMonteCarloSearch(useMonteCarloSearchOption);
    }
  }
  
  public void setNumberOfPlayers(int numberOfPlayers) {

    if (currentNumberOfPlayers != numberOfPlayers) {
      if (numberOfPlayers == ONE_PLAYER) {
        this.game.setOtherPlayer(
            new AlphaZeroPlayer(this.game.getOtherColor(),
                DEFAULT_MODEL_NAME,
                useMonteCarloSearch,
                0));
        this.currentNumberOfPlayers = ONE_PLAYER;
      } else if (numberOfPlayers == TWO_PLAYERS) {
        this.game.setPlayers(new HumanPlayer(PlaygroundConstants.YELLOW), new HumanPlayer(PlaygroundConstants.RED));
        this.currentNumberOfPlayers = TWO_PLAYERS;
      }
    }
  }

  public void play() {

    if (game.notOver()) {
      if (game.getCurrentPlayer() instanceof AlphaZeroPlayer) {
        game.getComputerPlayer().move();
      } else if (game.getOtherPlayer() instanceof AlphaZeroPlayer) {
        this.switchPlayers();
        game.getComputerPlayer().move();
      }
    }
  }

  protected void switchPlayers() {

    this.yellowPlayer.setColor(PlaygroundConstants.RED);
    this.redPlayer.setColor(PlaygroundConstants.YELLOW);
    AbstractPlayer oldYellowPlayer = this.yellowPlayer;
    this.yellowPlayer = this.redPlayer;
    this.redPlayer = oldYellowPlayer;
    this.game.setPlayers(this.yellowPlayer, this.redPlayer);
  }
}
