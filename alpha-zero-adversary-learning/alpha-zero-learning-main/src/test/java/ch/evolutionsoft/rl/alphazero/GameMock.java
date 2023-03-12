package ch.evolutionsoft.rl.alphazero;

import java.util.Set;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import ch.evolutionsoft.rl.Game;

public class GameMock extends Game {

  private static final INDArray staticBoard = Nd4j.zeros(2);

  private int staticGameEndAfterMoves = 2;
  private double staticGameEndValue = Game.MAX_WIN;
  private int moveNumber = 0;

  public GameMock(int moveNumber, double staticGameEndValue, int staticGameEndAfterMoves, int player) {
    
    super(player);
    this.moveNumber = moveNumber;
    this.staticGameEndValue = staticGameEndValue;
    this.staticGameEndAfterMoves = staticGameEndAfterMoves;
  }
  
  @Override
  public Game createNewInstance() {

    return new GameMock(this.moveNumber, this.staticGameEndValue, this.staticGameEndAfterMoves, this.currentPlayer);
  }
  
  @Override
  public int getNumberOfAllAvailableMoves() {

    return 2;
  }

  @Override
  public int getNumberOfCurrentMoves() {

    return 2;
  }

  @Override
  public INDArray getInitialBoard() {

    return staticBoard;
  }

  @Override
  public INDArray doFirstMove(int moveIndex) {

    return staticBoard;
  }

  @Override
  public Set<Integer> getValidMoveIndices() {
    
    return Set.of(0, 1);
  }

  @Override
  public INDArray getValidMoves() {

    return Nd4j.ones(2);
  }

  @Override
  public boolean gameEnded() {

    return this.moveNumber >= staticGameEndAfterMoves;
  }

  @Override
  public INDArray makeMove(int moveIndex, int player) {

    this.currentPlayer = getOtherPlayer(this.currentPlayer);
    this.moveNumber++;
    return staticBoard;
  }

  @Override
  public double getEndResult(int lastPlayer) {

    return this.staticGameEndValue;
  }

}
