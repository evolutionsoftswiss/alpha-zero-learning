package ch.evolutionsoft.rl.alphazero;

import java.util.Set;

import org.nd4j.linalg.api.ndarray.INDArray;

import ch.evolutionsoft.rl.Game;

public class TestGame extends Game {

  public TestGame() {
  }

  public TestGame(int currentPlayer) {
    super(currentPlayer);
  }

  @Override
  public int getNumberOfAllAvailableMoves() {
    return 7;
  }

  @Override
  public int getNumberOfCurrentMoves() {
    return 5;
  }

  @Override
  public INDArray getInitialBoard() {
    return null;
  }

  @Override
  public INDArray doFirstMove(int moveIndex) {
    return null;
  }

  @Override
  public Set<Integer> getValidMoveIndices() {
    return null;
  }

  @Override
  public INDArray getValidMoves() {
    return null;
  }

  @Override
  public boolean gameEnded() {
    return false;
  }

  @Override
  public INDArray makeMove(int moveIndex, int player) {
    return null;
  }

  @Override
  public double getEndResult(int lastPlayer) {
    return 0;
  }

}
