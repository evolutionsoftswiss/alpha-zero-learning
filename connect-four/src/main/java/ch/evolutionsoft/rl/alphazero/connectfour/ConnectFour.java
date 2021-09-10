package ch.evolutionsoft.rl.alphazero.connectfour;

import java.util.Set;

import org.nd4j.linalg.api.ndarray.INDArray;

import ch.evolutionsoft.rl.Game;

public class ConnectFour extends Game {

  public ConnectFour() {
    // TODO Auto-generated constructor stub
  }

  public ConnectFour(int currentPlayer) {
    super(currentPlayer);
    // TODO Auto-generated constructor stub
  }

  public ConnectFour(int currentPlayer, INDArray currentBoard) {
    super(currentPlayer, currentBoard);
    // TODO Auto-generated constructor stub
  }

  @Override
  public int getNumberOfAllAvailableMoves() {
    // TODO Auto-generated method stub
    return 0;
  }

  @Override
  public int getNumberOfCurrentMoves() {
    // TODO Auto-generated method stub
    return 0;
  }

  @Override
  public INDArray getInitialBoard() {
    // TODO Auto-generated method stub
    return null;
  }

  @Override
  public INDArray doFirstMove(int moveIndex) {
    // TODO Auto-generated method stub
    return null;
  }

  @Override
  public Set<Integer> getValidMoveIndices() {
    // TODO Auto-generated method stub
    return null;
  }

  @Override
  public INDArray getValidMoves() {
    // TODO Auto-generated method stub
    return null;
  }

  @Override
  public boolean gameEnded() {
    // TODO Auto-generated method stub
    return false;
  }

  @Override
  public INDArray makeMove(int moveIndex, int player) {
    // TODO Auto-generated method stub
    return null;
  }

  @Override
  public double getEndResult(int lastPlayer) {
    // TODO Auto-generated method stub
    return 0;
  }

}
