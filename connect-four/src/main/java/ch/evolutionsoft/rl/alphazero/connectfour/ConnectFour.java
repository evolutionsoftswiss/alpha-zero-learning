package ch.evolutionsoft.rl.alphazero.connectfour;

import static ch.evolutionsoft.rl.alphazero.connectfour.playground.PlaygroundConstants.*;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import ch.evolutionsoft.rl.AdversaryLearningConstants;
import ch.evolutionsoft.rl.AdversaryTrainingExample;
import ch.evolutionsoft.rl.Game;
import ch.evolutionsoft.rl.alphazero.connectfour.playground.BinaryPlayground;

public class BinaryConnectFour extends Game {

  public static final int CURRENT_PLAYER_CHANNEL = 2;
  public static final int NUMBER_OF_BOARD_CHANNELS = 3;
  
  public static final INDArray ZEROS_PLAYGROUND_IMAGE = Nd4j.zeros(ROW_COUNT, COLUMN_COUNT);
  public static final INDArray ONES_PLAYGROUND_IMAGE = Nd4j.ones(ROW_COUNT, COLUMN_COUNT);
  public static final INDArray MINUS_ONES_PLAYGROUND_IMAGE = ZEROS_PLAYGROUND_IMAGE.sub(1);
  
  public static final INDArray EMPTY_CONVOLUTIONAL_PLAYGROUND = Nd4j.create(3, ROW_COUNT, COLUMN_COUNT);
  static {
    EMPTY_CONVOLUTIONAL_PLAYGROUND.putSlice(YELLOW, ZEROS_PLAYGROUND_IMAGE);
    EMPTY_CONVOLUTIONAL_PLAYGROUND.putSlice(RED, ZEROS_PLAYGROUND_IMAGE);
    EMPTY_CONVOLUTIONAL_PLAYGROUND.putSlice(EMPTY, ONES_PLAYGROUND_IMAGE);
  }
  
  BinaryPlayground arrayPlayground = new BinaryPlayground();
  int lastMoveColumn = -1;

  public BinaryConnectFour() {
    
  }

  public BinaryConnectFour(int currentPlayer) {

    super(currentPlayer);
  }
  
  @Override
  public Game createNewInstance() {
    
    BinaryConnectFour connectFour = new BinaryConnectFour(this.currentPlayer);
    
    long[] boardPosition = new long[2];
    System.arraycopy(this.arrayPlayground.getPosition(), 0, boardPosition, 0, boardPosition.length);
    int[] columnHeights = new int[COLUMN_COUNT];
    System.arraycopy(this.arrayPlayground.getColumnHeights(), 0, columnHeights, 0, columnHeights.length);
    connectFour.arrayPlayground = new BinaryPlayground(boardPosition, columnHeights, 42);
    
    connectFour.lastMoveColumn = this.lastMoveColumn;
    connectFour.currentBoard = this.currentBoard.dup();
    
    return connectFour;
  }
  
  public Game createNewInstance(BinaryPlayground arrayPlayground) {
    
    BinaryConnectFour connectFour = new BinaryConnectFour();

    long[] boardPosition = new long[2];
    System.arraycopy(arrayPlayground.getPosition(), 0, boardPosition, 0, boardPosition.length);
    int[] columnHeights = new int[COLUMN_COUNT];
    System.arraycopy(arrayPlayground.getColumnHeights(), 0, columnHeights, 0, columnHeights.length);
    connectFour.arrayPlayground = new BinaryPlayground(boardPosition, columnHeights, arrayPlayground.getFieldsLeft());
    
    connectFour.lastMoveColumn = -1;
    int yellowStones = 0;
    int redStones = 0;

    for (int row = 0; row < ROW_COUNT; row++) {
        for (int column = 0; column < COLUMN_COUNT; column++) {
          
          long mask = 1L << column * BinaryPlayground.BITS_PER_COLUMN + row;
          int conversedRowIndex = ROW_COUNT - row - 1;

          if ((boardPosition[0] & mask) != 0) {
            connectFour.currentBoard.putScalar(
                YELLOW,
                conversedRowIndex,
                column,
                AdversaryLearningConstants.ONE);
            yellowStones++;
          }
          else if ((boardPosition[1] & mask) != 0) {
            
            connectFour.currentBoard.putScalar(
                RED,
                conversedRowIndex,
                column,
                AdversaryLearningConstants.ONE);
            redStones++;
          }
        }
    }
    if (yellowStones == redStones) {

      connectFour.currentBoard.putSlice(
          CURRENT_PLAYER_CHANNEL,
          ONES_PLAYGROUND_IMAGE); 
    
    } else {

      connectFour.currentBoard.putSlice(
          CURRENT_PLAYER_CHANNEL,
          MINUS_ONES_PLAYGROUND_IMAGE);
      connectFour.currentPlayer = Game.MIN_PLAYER;
    }
    
    return connectFour;
  }

  @Override
  public int getNumberOfAllAvailableMoves() {

    return COLUMN_COUNT;
  }

  @Override
  public int getNumberOfCurrentMoves() {

    return COLUMN_COUNT;
  }

  @Override
  public INDArray getInitialBoard() {

    return EMPTY_CONVOLUTIONAL_PLAYGROUND.dup();
  }

  @Override
  public INDArray doFirstMove(int moveIndex) {
 
    return makeMove(moveIndex, Game.MAX_PLAYER);
  }

  @Override
  public Set<Integer> getValidMoveIndices() {

    Set<Integer> emptyFieldsIndices = new HashSet<>(10);
    
    for (int column = 0; column < COLUMN_COUNT; column++) {
  
      if (AdversaryLearningConstants.ZERO == this.currentBoard.getDouble(YELLOW, 0, column) &&
          AdversaryLearningConstants.ZERO == this.currentBoard.getDouble(RED, 0, column)) {
  
        emptyFieldsIndices.add(column);
      }
    }
    
    return emptyFieldsIndices;
  }

  @Override
  public INDArray getValidMoves() {

    INDArray validMoves = Nd4j.zeros(COLUMN_COUNT);
    
    Set<Integer> validMoveIndices = getValidMoveIndices();
    validMoveIndices.stream().forEach(index -> validMoves.putScalar(index, AdversaryLearningConstants.ONE));
    
    return validMoves;
  }

  @Override
  public boolean gameEnded() {

    int otherConnectFourPlayer = this.convertGamePlayerToConnectFourPlayer(
        this.getOtherPlayer(this.currentPlayer)
        );

    if (YELLOW == otherConnectFourPlayer) {
      
      return this.arrayPlayground.fourInARow(this.arrayPlayground.getFirstPlayerPosition()) ||
                  this.arrayPlayground.getAvailableColumns().isEmpty();
    }
    
    return this.arrayPlayground.fourInARow(this.arrayPlayground.getSecondPlayerPosition()) ||
                this.arrayPlayground.getAvailableColumns().isEmpty();
  }

  @Override
  public INDArray makeMove(int moveIndex, int player) {

    int connectFourPlayer = convertGamePlayerToConnectFourPlayer(player);
    int playedRow = this.arrayPlayground.trySetField(moveIndex, connectFourPlayer);
    this.lastMoveColumn = moveIndex;
 
    this.currentBoard = this.currentBoard.dup();
    if (Game.MAX_PLAYER == player) {

      this.currentBoard.putSlice(CURRENT_PLAYER_CHANNEL, ONES_PLAYGROUND_IMAGE); 
    } else {

      this.currentBoard.putSlice(CURRENT_PLAYER_CHANNEL, MINUS_ONES_PLAYGROUND_IMAGE);
    }

    this.currentBoard.putScalar(connectFourPlayer, ROW_COUNT - playedRow - 1L, moveIndex, AdversaryLearningConstants.ONE);

    this.currentPlayer = getOtherPlayer(this.currentPlayer);
    
    return this.currentBoard;
  }

  @Override
  public double getEndResult(int lastPlayer) {

    boolean maxWin = arrayPlayground.fourInARow(this.arrayPlayground.getFirstPlayerPosition());
    
    if (maxWin) {
      
      return Game.MAX_WIN;
    }
    
    boolean minWin = arrayPlayground.fourInARow(this.arrayPlayground.getSecondPlayerPosition());
    
    if (minWin) {
      
      return Game.MIN_WIN;
    
    }
    
    return Game.DRAW;
    
  }

  @Override
  public List<AdversaryTrainingExample> getSymmetries(INDArray board, INDArray actionProbabilities, int currentPlayer,
      int iteration) {

    List<AdversaryTrainingExample> symmetries = new ArrayList<>();

    INDArray actionMirrorHorizontal = Nd4j.reverse(actionProbabilities.dup());
    INDArray newPlaygroundMirrorHorizontal = mirrorBoardVertically(board.dup());
    symmetries.add(
        new AdversaryTrainingExample(
            newPlaygroundMirrorHorizontal,
            currentPlayer,
            actionMirrorHorizontal,
            iteration)
        );
    
    return symmetries;
  }

  static INDArray mirrorBoardVertically(INDArray playgroundRotation) {

    INDArray boardPlayerMirrorHorizontal = playgroundRotation.slice(2);
    INDArray maxPlayerMirrorHorizontal = mirrorBoardPartVertically(playgroundRotation.slice(0));
    INDArray minPlayerMirrorHorizontal = mirrorBoardPartVertically(playgroundRotation.slice(1));
    
    return Nd4j.create(
        List.of(maxPlayerMirrorHorizontal, minPlayerMirrorHorizontal, boardPlayerMirrorHorizontal),
        new long[] {3, 6, 7});
  }
  
  static INDArray mirrorBoardPartVertically(INDArray toMirror) {

    List<INDArray> verticalMirrors = new LinkedList<>();
    for (int row = 0; row < toMirror.shape()[0]; row++) {
      
      verticalMirrors.add(Nd4j.reverse(toMirror.getRow(row)));
    }
    
    return Nd4j.create(verticalMirrors,
        new long[] {6, 7});
  }

  int convertGamePlayerToConnectFourPlayer(int gamePlayer) {
 
    return gamePlayer - 1;
  }
}
