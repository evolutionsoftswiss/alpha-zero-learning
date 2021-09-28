package ch.evolutionsoft.rl.tictactoe;

import static ch.evolutionsoft.rl.tictactoe.TicTacToeConstants.*;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ch.evolutionsoft.rl.AdversaryLearningConstants;
import ch.evolutionsoft.rl.AdversaryTrainingExample;
import ch.evolutionsoft.rl.Game;

/**
 * Initial board setup of the 3x3x3 INDArray.
 * [
 *  [ // Current player max has all 1, current player min has all -1
 *   [    1.0000,    1.0000,    1.0000], 
 *   [    1.0000,    1.0000,    1.0000], 
 *   [    1.0000,    1.0000,    1.0000]
 *  ], 
 *  [ // Max stones X's
 *   [         0,         0,         0], 
 *   [         0,         0,         0], 
 *   [         0,         0,         0]
 *  ],
 *  [ // Min stones O's
 *   [         0,         0,         0], 
 *   [         0,         0,         0], 
 *   [         0,         0,         0]
 *  ]
 * ]
 * 
 * @author evolutionsoft
 */
public class TicTacToe extends Game {

  private static final Logger log = LoggerFactory.getLogger(TicTacToe.class);
  
  public TicTacToe(int currentPlayer) {

    super(currentPlayer);
  }

  @Override
  public Game createNewInstance() {

    TicTacToe ticTacToe = new TicTacToe(currentPlayer); 
    ticTacToe.currentBoard = this.currentBoard.dup();
    
    return ticTacToe;
  }
  
  @Override
  public int getNumberOfAllAvailableMoves() {

    return TicTacToeConstants.COLUMN_COUNT;
  }

  @Override
  public int getNumberOfCurrentMoves() {

    return TicTacToeConstants.COLUMN_COUNT;
  }

  /**
   * Create additional symmetric {@link AdversaryTrainingExample} with the given values.
   * Adds all symmetries obtained by horizontal mirroring and rotating 90 degrees.
   * The seven symmetries in the returned List may be identical.
   * 
   * @param board a 3x3x3 INDArray for the board state in TicTacToe
   * @param actionProbabilities 1x9 INDArray for the current move index probabilities
   * @param currentPlayer the current player of the created {@link AdversaryTrainingExample}
   * @param iteration the current iteration, may identify the recentness of an {@link AdversaryTrainingExample}
   */
  @Override  
  public List<AdversaryTrainingExample> getSymmetries(INDArray board, INDArray actionProbabilities, int currentPlayer, int iteration) {
    
    List<AdversaryTrainingExample> symmetries = new ArrayList<>();
    
    INDArray twoDimensionalActionProbabilities = actionProbabilities.reshape(IMAGE_SIZE, IMAGE_SIZE);
    INDArray playgroundRotation = Nd4j.create(IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE);
    Nd4j.copy(board, playgroundRotation);

    INDArray actionMirrorHorizontal = mirrorBoardPartHorizontally(twoDimensionalActionProbabilities);
    actionMirrorHorizontal = actionMirrorHorizontal.reshape(COLUMN_COUNT);
    INDArray newPlaygroundMirrorHorizontal = mirrorBoardHorizontally(playgroundRotation);
    symmetries.add(
        new AdversaryTrainingExample(
            newPlaygroundMirrorHorizontal.dup(),
            currentPlayer,
            actionMirrorHorizontal,
            iteration)
        );
    
    for (int rotation = 1; rotation < 4; rotation++) {
     
      INDArray newActionRotation = rotate90(twoDimensionalActionProbabilities.dup());
      INDArray newPlaygroundRotation = rotateBoard90(playgroundRotation.dup());
      symmetries.add(
          new AdversaryTrainingExample(
              newPlaygroundRotation.dup(),
              currentPlayer,
              newActionRotation.reshape(COLUMN_COUNT).dup(),
              iteration)
          );

      INDArray newActionMirrorHorizontal = mirrorBoardPartHorizontally(newActionRotation.dup());
      newActionMirrorHorizontal = newActionMirrorHorizontal.reshape(COLUMN_COUNT);
      newPlaygroundMirrorHorizontal = mirrorBoardHorizontally(newPlaygroundRotation.dup());
      symmetries.add(
          new AdversaryTrainingExample(
              newPlaygroundMirrorHorizontal.dup(),
              currentPlayer,
              newActionMirrorHorizontal.dup(),
              iteration)
          );
      
      Nd4j.copy(newActionRotation, twoDimensionalActionProbabilities);
      Nd4j.copy(newPlaygroundRotation, playgroundRotation);
    }
    
    
    return symmetries;
  }

  @Override  
  public INDArray getInitialBoard() {
    
    return TicTacToeConstants.EMPTY_CONVOLUTIONAL_PLAYGROUND.dup();
  }

  @Override
  public INDArray doFirstMove(int moveIndex) {
    
    return makeMove(moveIndex, TicTacToeConstants.MAX_PLAYER_CHANNEL);
  }

  /**
   * TicTacToe game ended when no fields are empty or a player has won.
   */
  @Override
  public boolean gameEnded() {

    return getValidMoveIndices().isEmpty() ||
        getEndResult(-1) != 0.5;
  }

  @Override
  public double getEndResult(int lastPlayer) {

    boolean maxWin = horizontalWin(this.currentBoard, Game.MAX_PLAYER) || 
        verticalWin(this.currentBoard, Game.MAX_PLAYER) || 
        diagonalWin(this.currentBoard, Game.MAX_PLAYER);
    
    if (maxWin) {
      
      return 1.0;
    }
    
    boolean minWin = horizontalWin(this.currentBoard, Game.MIN_PLAYER) || 
        verticalWin(this.currentBoard, Game.MIN_PLAYER) || 
        diagonalWin(this.currentBoard, Game.MIN_PLAYER);
    
    if (minWin) {
      
      return 0.0;
    
    }
    
    return 0.5;
  }

  @Override
  public INDArray makeMove(int moveIndex, int player) {

    int row = moveIndex / IMAGE_SIZE;
    int column = moveIndex % IMAGE_SIZE;
    
    INDArray newBoard = this.currentBoard.dup();
    if (MIN_PLAYER_CHANNEL == player) {

      newBoard.putRow(CURRENT_PLAYER_CHANNEL, ONES_PLAYGROUND_IMAGE); 
    } else {

      newBoard.putRow(CURRENT_PLAYER_CHANNEL, MINUS_ONES_PLAYGROUND_IMAGE);
    }
    newBoard.putScalar(player, row, column, OCCUPIED_IMAGE_POINT);

    this.currentBoard = newBoard;
    this.currentPlayer = getOtherPlayer(this.currentPlayer);
    
    return newBoard.dup();
  }

  /**
   * In TicTacToe the valid moves are all empty fields.
   */
  @Override
  public Set<Integer> getValidMoveIndices() {
    
    Set<Integer> emptyFieldsIndices = new HashSet<>(SMALL_CAPACITY);
    
    for (int row = 0; row < IMAGE_SIZE; row++) {
      for (int column = 0; column < IMAGE_SIZE; column++) {
  
        if (AdversaryLearningConstants.ZERO == this.currentBoard.getDouble(MAX_PLAYER_CHANNEL, row, column) &&
            AdversaryLearningConstants.ZERO == this.currentBoard.getDouble(MIN_PLAYER_CHANNEL, row, column)) {
  
          emptyFieldsIndices.add(IMAGE_SIZE * row + column);
        }
      }
    }
    
    return emptyFieldsIndices;
  }

  @Override
  public INDArray getValidMoves() {
    
    INDArray validMoves = Nd4j.zeros(COLUMN_COUNT);
    
    for (int row = 0; row < IMAGE_SIZE; row++) {
      for (int column = 0; column < IMAGE_SIZE; column++) {
  
        if (AdversaryLearningConstants.ZERO == this.currentBoard.getDouble(MAX_PLAYER_CHANNEL, row, column) &&
            AdversaryLearningConstants.ZERO == this.currentBoard.getDouble(MIN_PLAYER_CHANNEL, row, column)) {
          
          validMoves.putScalar(IMAGE_SIZE * (long) row + column, AdversaryLearningConstants.ONE);
        }
      }
    }
    
    return validMoves;
  }

  /**
   * Do an evaluation against labels fom supervised learning.
   * Only an indication for improvement during training, as symmetry equivalent moves
   * or other correct moves may be learned different to the labels.
   */
  @Override
  public void evaluateNetwork(ComputationGraph computationGraph) {

    EvaluationMain.evaluateNetwork(computationGraph);
  }

  /**
   * Log the move probabilities for a few moves during training.
   */
  @Override
  public void evaluateBoardActionExamples(ComputationGraph convolutionalNetwork) {

    INDArray[] centerFieldOpeningAnswer = convolutionalNetwork.output(generateCenterFieldInputImagesConvolutional());
    INDArray[] cornerFieldOpeningAnswer = convolutionalNetwork
        .output(generateLastCornerFieldInputImagesConvolutional());
    INDArray[] fieldOneOpeningAnswer = convolutionalNetwork
        .output(generateFieldOneInputImagesConvolutional());
    INDArray[] fieldOneCenterTwoOpeningAnswer = convolutionalNetwork
        .output(generateFieldOneCenterAndTwoThreatConvolutional());
    INDArray emptyFieldBatch = Nd4j.create(1, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE);
    emptyFieldBatch.putRow(0, TicTacToeConstants.EMPTY_CONVOLUTIONAL_PLAYGROUND);
    INDArray[] emptyFieldProbs = convolutionalNetwork.output(emptyFieldBatch);
    
    log.info("Answer to center field opening: {}\nValue: {}", centerFieldOpeningAnswer[0], centerFieldOpeningAnswer[1]);
    log.info("Answer to last corner field opening: {}\nValue: {}", cornerFieldOpeningAnswer[0], cornerFieldOpeningAnswer[1]);
    log.info("Answer to field one, center and two threat: {}\nValue: {}", fieldOneCenterTwoOpeningAnswer[0], fieldOneCenterTwoOpeningAnswer[1]);
    log.info("Answer to field one opening: {}\nValue: {}", fieldOneOpeningAnswer[0], fieldOneOpeningAnswer[1]);
    log.info("Opening probailities: {}\nValue: {}", emptyFieldProbs[0], emptyFieldProbs[1]);
  }

  INDArray generateCenterFieldInputImagesConvolutional() {

    INDArray middleFieldMove = TicTacToeConstants.EMPTY_CONVOLUTIONAL_PLAYGROUND.dup();
    INDArray emptyImage1 = Nd4j.ones(1, IMAGE_SIZE, IMAGE_SIZE).mul(-1);
    middleFieldMove.putRow(0, emptyImage1);
    middleFieldMove.putScalar(1, 1, 1, OCCUPIED_IMAGE_POINT);
    INDArray graphSingleBatchInput1 = Nd4j.create(1, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE);
    graphSingleBatchInput1.putRow(0, middleFieldMove);
    return graphSingleBatchInput1;
  }

  INDArray generateLastCornerFieldInputImagesConvolutional() {

    INDArray cornerFieldMove = TicTacToeConstants.EMPTY_CONVOLUTIONAL_PLAYGROUND.dup();
    INDArray emptyImage2 = Nd4j.ones(1, IMAGE_SIZE, IMAGE_SIZE).mul(-1);
    cornerFieldMove.putRow(0, emptyImage2);
    cornerFieldMove.putScalar(1, 2, 2, OCCUPIED_IMAGE_POINT);
    INDArray graphSingleBatchInput2 = Nd4j.create(1, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE);
    graphSingleBatchInput2.putRow(0, cornerFieldMove);
    return graphSingleBatchInput2;
  }

  INDArray generateFieldOneInputImagesConvolutional() {

    INDArray fieldOneMaxMove = TicTacToeConstants.EMPTY_CONVOLUTIONAL_PLAYGROUND.dup();
    INDArray emptyImage1 = Nd4j.ones(1, IMAGE_SIZE, IMAGE_SIZE).mul(-1);
    fieldOneMaxMove.putRow(0, emptyImage1);
    fieldOneMaxMove.putScalar(1, 0, 0, OCCUPIED_IMAGE_POINT);
    INDArray graphSingleBatchInput2 = Nd4j.create(1, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE);
    graphSingleBatchInput2.putRow(0, fieldOneMaxMove);
    return graphSingleBatchInput2;
  }

  INDArray generateFieldOneCenterAndTwoThreatConvolutional() {

    INDArray fieldOneCenterTwoMoves = TicTacToeConstants.EMPTY_CONVOLUTIONAL_PLAYGROUND.dup();
    INDArray emptyImage1 = Nd4j.ones(1, IMAGE_SIZE, IMAGE_SIZE).mul(-1);
    fieldOneCenterTwoMoves.putRow(0, emptyImage1);
    fieldOneCenterTwoMoves.putScalar(1, 0, 0, OCCUPIED_IMAGE_POINT);
    fieldOneCenterTwoMoves.putScalar(2, 1, 1, OCCUPIED_IMAGE_POINT);
    fieldOneCenterTwoMoves.putScalar(1, 0, 1, OCCUPIED_IMAGE_POINT);
    INDArray graphSingleBatchInput2 = Nd4j.create(1, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE);
    graphSingleBatchInput2.putRow(0, fieldOneCenterTwoMoves);
    return graphSingleBatchInput2;
  }

  public String toString() {
    
    return "player: " + this.currentPlayer + System.lineSeparator() + this.currentBoard;
  }

  static boolean horizontalWin(INDArray board, int player) {

    return (board.getDouble(player, 0, 0) == OCCUPIED_IMAGE_POINT &&
        board.getDouble(player, 0, 1) == OCCUPIED_IMAGE_POINT &&
            board.getDouble(player, 0, 2) == OCCUPIED_IMAGE_POINT) ||
           (board.getDouble(player, 1, 0) == OCCUPIED_IMAGE_POINT &&
               board.getDouble(player, 1, 1) == OCCUPIED_IMAGE_POINT &&
                   board.getDouble(player, 1, 2) == OCCUPIED_IMAGE_POINT) ||
           (board.getDouble(player, 2, 0) == OCCUPIED_IMAGE_POINT &&
               board.getDouble(player, 2, 1) == OCCUPIED_IMAGE_POINT &&
                   board.getDouble(player, 2, 2) == OCCUPIED_IMAGE_POINT);
  }

  static boolean diagonalWin(INDArray board, int player) {

    return (board.getDouble(player, 0, 0) == OCCUPIED_IMAGE_POINT &&
        board.getDouble(player, 1, 1) == OCCUPIED_IMAGE_POINT &&
            board.getDouble(player, 2, 2) == OCCUPIED_IMAGE_POINT) ||
           (board.getDouble(player, 0, 2) == OCCUPIED_IMAGE_POINT &&
               board.getDouble(player, 1, 1) == OCCUPIED_IMAGE_POINT &&
                   board.getDouble(player, 2, 0) == OCCUPIED_IMAGE_POINT);
  }

  static boolean verticalWin(INDArray board, int player) {

    return (board.getDouble(player, 0, 0) == OCCUPIED_IMAGE_POINT &&
        board.getDouble(player, 1, 0) == OCCUPIED_IMAGE_POINT &&
            board.getDouble(player, 2, 0) == OCCUPIED_IMAGE_POINT) ||
           (board.getDouble(player, 0, 1) == OCCUPIED_IMAGE_POINT &&
               board.getDouble(player, 1, 1) == OCCUPIED_IMAGE_POINT &&
                   board.getDouble(player, 2, 1) == OCCUPIED_IMAGE_POINT) ||
           (board.getDouble(player, 0, 2) == OCCUPIED_IMAGE_POINT &&
               board.getDouble(player, 1, 2) == OCCUPIED_IMAGE_POINT &&
                   board.getDouble(player, 2, 2) == OCCUPIED_IMAGE_POINT);
  }

  static INDArray mirrorBoardHorizontally(INDArray playgroundRotation) {

    INDArray boardPlayerMirrorHorizontal = playgroundRotation.slice(CURRENT_PLAYER_CHANNEL);
    INDArray maxPlayerMirrorHorizontal = mirrorBoardPartHorizontally(playgroundRotation.slice(MAX_PLAYER_CHANNEL));
    INDArray minPlayerMirrorHorizontal = mirrorBoardPartHorizontally(playgroundRotation.slice(MIN_PLAYER_CHANNEL));
    
    return createNewBoard(boardPlayerMirrorHorizontal, maxPlayerMirrorHorizontal,
        minPlayerMirrorHorizontal);
  }
  
  static INDArray mirrorBoardPartHorizontally(INDArray toMirror) {
    
    INDArray mirrorHorizontal = Nd4j.ones(toMirror.shape()).neg();
    mirrorHorizontal.putRow(0, toMirror.slice(2));
    mirrorHorizontal.putRow(1, toMirror.slice(1));
    mirrorHorizontal.putRow(2, toMirror.slice(0));
    
    return mirrorHorizontal;
  }

  static INDArray rotateBoard90(INDArray playgroundRotation) {

    INDArray boardEmptyRotation = playgroundRotation.slice(CURRENT_PLAYER_CHANNEL);
    INDArray maxPlayerRotation = rotate90(playgroundRotation.slice(MAX_PLAYER_CHANNEL));
    INDArray minPlayerRotation = rotate90(playgroundRotation.slice(MIN_PLAYER_CHANNEL));
    
    return createNewBoard(boardEmptyRotation, maxPlayerRotation, minPlayerRotation);
  }

  static INDArray createNewBoard(INDArray newEmptyBoardPart, INDArray newMaxPlayerBoardPart, INDArray newMinPlayerBoardPart) {

    INDArray newPlaygroundRotation = Nd4j.create(IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE);
    newPlaygroundRotation.putRow(CURRENT_PLAYER_CHANNEL, newEmptyBoardPart);
    newPlaygroundRotation.putRow(MAX_PLAYER_CHANNEL, newMaxPlayerBoardPart);
    newPlaygroundRotation.putRow(MIN_PLAYER_CHANNEL, newMinPlayerBoardPart);

    return newPlaygroundRotation;
  }
  
  static INDArray rotate90(INDArray toRotate) {
    
    INDArray rotated90 = Nd4j.ones(toRotate.shape());
    
    for (int col = 0; col < toRotate.shape()[1]; col++) {
     
      INDArray slice = toRotate.getColumn(col).dup();
      rotated90.putRow(col, Nd4j.reverse(slice));
    } 
    return rotated90;
  }
}
