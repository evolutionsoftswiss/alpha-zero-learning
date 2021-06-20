package ch.evolutionsoft.rl.tictactoe;

import static ch.evolutionsoft.net.game.NeuralNetConstants.DOUBLE_COMPARISON_EPSILON;
import static ch.evolutionsoft.net.game.tictactoe.TicTacToeConstants.*;
import static ch.evolutionsoft.net.game.tictactoe.TicTacToeGameHelper.equalsEpsilon;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ch.evolutionsoft.net.game.tictactoe.TicTacToeConstants;
import ch.evolutionsoft.rl.AdversaryTrainingExample;
import ch.evolutionsoft.rl.Game;

public class TicTacToe extends Game {
  
  private static final Logger log = LoggerFactory.getLogger(TicTacToe.class);
  
  public static final int[] COLUMN_INDICES = new int[] {0, 1, 2, 3, 4, 5, 6, 7, 8};
  

  @Override
  public int getFieldCount() {
    
    return TicTacToeConstants.COLUMN_COUNT;
  }

  @Override  
  public List<AdversaryTrainingExample> getSymmetries(INDArray playground, INDArray actionProbabilities, int currentPlayer, int iteration) {
    
    List<AdversaryTrainingExample> symmetries = new ArrayList<>();
    
    INDArray twoDimensionalActionProbabilities = actionProbabilities.reshape(IMAGE_SIZE, IMAGE_SIZE);
    INDArray playgroundRotation = Nd4j.create(IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE);
    Nd4j.copy(playground, playgroundRotation);

    INDArray actionMirrorVertical = mirrorVertical(twoDimensionalActionProbabilities.dup());
    actionMirrorVertical = actionMirrorVertical.reshape(COLUMN_COUNT);
    INDArray boardMirrorVertical = mirrorBoardVertically(playgroundRotation.dup());
    symmetries.add(
        new AdversaryTrainingExample(
            boardMirrorVertical.dup(),
            currentPlayer,
            actionMirrorVertical,
            iteration)
        );
    
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
      
      INDArray newActionMirrorVertical = mirrorVertical(newActionRotation.dup());
      INDArray newPlaygroundMirror = mirrorBoardVertically(newPlaygroundRotation.dup());
      symmetries.add(
          new AdversaryTrainingExample(
              newPlaygroundMirror.dup(),
              currentPlayer,
              newActionMirrorVertical.reshape(COLUMN_COUNT).dup(),
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
    
    return TicTacToeConstants.EMPTY_CONVOLUTIONAL_PLAYGROUND;
  }

  @Override
  public INDArray doFirstMove(int index) {

    INDArray emptyBoard = TicTacToeConstants.EMPTY_CONVOLUTIONAL_PLAYGROUND.dup();
    
    INDArray newBoard = makeMove(emptyBoard, index, TicTacToeConstants.MAX_PLAYER_CHANNEL);
    
    return newBoard;
  }

  @Override
  public boolean gameEnded(INDArray board) {

    return getEmptyFields(board).isEmpty() ||
        hasWon(board, MAX_PLAYER_CHANNEL) ||
        hasWon(board, MIN_PLAYER_CHANNEL);
  }

  @Override
  public boolean hasWon(INDArray board, int player) {

    return horizontalWin(board, player) || verticalWin(board, player) || diagonalWin(board, player);
  }

  @Override
  public INDArray makeMove(INDArray board, int flatIndex, int player) {

    int row = flatIndex / IMAGE_CHANNELS;
    int column = flatIndex % IMAGE_CHANNELS;
    
    INDArray newBoard = board.dup();
    if (MIN_PLAYER_CHANNEL == player) {

      newBoard.putRow(PLAYER_CHANNEL, ONES_PLAYGROUND_IMAGE); 
    } else {

      newBoard.putRow(PLAYER_CHANNEL, MINUS_ONES_PLAYGROUND_IMAGE);
    }
    newBoard.putScalar(player, row, column, OCCUPIED_IMAGE_POINT);

    return newBoard;
  }

  @Override
  public Set<Integer> getEmptyFields(INDArray playground) {
    
    Set<Integer> emptyFieldsIndices = new HashSet<>(SMALL_CAPACITY);
    
    for (int row = 0; row < IMAGE_SIZE; row++) {
      for (int column = 0; column < IMAGE_SIZE; column++) {
  
        if (equalsEpsilon(EMPTY_IMAGE_POINT,
            playground.getDouble(MAX_PLAYER_CHANNEL, row, column),
            DOUBLE_COMPARISON_EPSILON) &&
            equalsEpsilon(EMPTY_IMAGE_POINT,
                playground.getDouble(MIN_PLAYER_CHANNEL, row, column),
                DOUBLE_COMPARISON_EPSILON)) {
  
          emptyFieldsIndices.add(IMAGE_SIZE * row + column);
        }
      }
    }
    
    return emptyFieldsIndices;
  }

  @Override
  public int getOtherPlayer(Set<Integer> emptyFields) {
    
    boolean evenOccupiedFields = 0 == (COLUMN_COUNT - emptyFields.size()) % 2;
    
    if (evenOccupiedFields) {
      
      return MIN_PLAYER_CHANNEL;
    }
    
    return MAX_PLAYER_CHANNEL;
  }

  @Override
  public int getOtherPlayer(int color) {
    
    if (TicTacToeConstants.MAX_PLAYER_CHANNEL == color) {
      
      return MIN_PLAYER_CHANNEL;
    }
    
    return MAX_PLAYER_CHANNEL;
  }

  @Override
  public INDArray getValidMoves(INDArray playground) {
    
    INDArray validMoves = Nd4j.zeros(COLUMN_COUNT);
    
    for (int row = 0; row < IMAGE_SIZE; row++) {
      for (int column = 0; column < IMAGE_SIZE; column++) {
  
        if (equalsEpsilon(EMPTY_IMAGE_POINT,
            playground.getDouble(MAX_PLAYER_CHANNEL, row, column),
            DOUBLE_COMPARISON_EPSILON) && 
            equalsEpsilon(EMPTY_IMAGE_POINT,
                playground.getDouble(MIN_PLAYER_CHANNEL, row, column),
                DOUBLE_COMPARISON_EPSILON) ) {
          
          validMoves.putScalar(IMAGE_SIZE * row + column, 1f);
        }
      }
    }
    
    return validMoves;
  }
  
  @Override
  public void evaluateNetwork(ComputationGraph computationGraph) {

    EvaluationMain.evaluateNetwork(computationGraph);
  }

  @Override
  public void evaluateOpeningAnswers(ComputationGraph convolutionalNetwork) {

    INDArray centerFieldOpeningAnswer = convolutionalNetwork.output(generateCenterFieldInputImagesConvolutional())[0];
    INDArray cornerFieldOpeningAnswer = convolutionalNetwork
        .output(generateLastCornerFieldInputImagesConvolutional())[0];
    INDArray fieldOneOpeningAnswer = convolutionalNetwork
        .output(generateFieldOneInputImagesConvolutional())[0];
    INDArray fieldOneCenterTwoOpeningAnswer = convolutionalNetwork
        .output(generateFieldOneCenterAndTwoThreatConvolutional())[0];

    log.info("Answer to center field opening: {}", centerFieldOpeningAnswer);
    log.info("Answer to last corner field opening: {}", cornerFieldOpeningAnswer);
    log.info("Answer to field one, center and two threat: {}", fieldOneCenterTwoOpeningAnswer);
    log.info("Answer to field one opening: {}", fieldOneOpeningAnswer);
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
  
  public int getCurrentPlayer(Set<Integer> emptyFields) {
    
    boolean oddOccupiedFields = 1 == (COLUMN_COUNT - emptyFields.size()) % 2;
    
    if (oddOccupiedFields) {
      
      return MIN_PLAYER_CHANNEL;
    }
    
    return MAX_PLAYER_CHANNEL;
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

    INDArray boardEmptyMirrorHorizontal = playgroundRotation.slice(PLAYER_CHANNEL);
    INDArray maxPlayerMirrorHorizontal = mirrorBoardPartHorizontally(playgroundRotation.slice(MAX_PLAYER_CHANNEL));
    INDArray minPlayerMirrorHorizontal = mirrorBoardPartHorizontally(playgroundRotation.slice(MIN_PLAYER_CHANNEL));
    
    INDArray newPlaygroundMirrorHorizontal = createNewBoard(boardEmptyMirrorHorizontal, maxPlayerMirrorHorizontal,
        minPlayerMirrorHorizontal);

    return newPlaygroundMirrorHorizontal;
  }
  
  static INDArray mirrorBoardPartHorizontally(INDArray toMirror) {
    
    INDArray mirrorHorizontal = Nd4j.ones(toMirror.shape()).neg();
    mirrorHorizontal.putRow(0, toMirror.slice(2));
    mirrorHorizontal.putRow(1, toMirror.slice(1));
    mirrorHorizontal.putRow(2, toMirror.slice(0));
    
    return mirrorHorizontal;
  }
  
  static INDArray mirrorBoardVertically(INDArray boardToMirror) {

    INDArray boardEmptyMirror = boardToMirror.slice(PLAYER_CHANNEL);
    INDArray maxPlayerMirror = mirrorVertical(boardToMirror.slice(MAX_PLAYER_CHANNEL));
    INDArray minPlayerMirror = mirrorVertical(boardToMirror.slice(MIN_PLAYER_CHANNEL));
    
    INDArray boardMirrorVertical = createNewBoard(boardEmptyMirror, maxPlayerMirror, minPlayerMirror);
    
    return boardMirrorVertical;
  }
  
  static INDArray mirrorVertical(INDArray toMirror) {
    
    INDArray mirroredVertical = Nd4j.ones(toMirror.shape()).neg();
    
    for (int row = 0; row < toMirror.shape()[0]; row++) {
      
      mirroredVertical.putRow(row, Nd4j.reverse(toMirror.getRow(row)));
    }
    
    return mirroredVertical;
  }

  static INDArray rotateBoard90(INDArray playgroundRotation) {

    INDArray boardEmptyRotation = playgroundRotation.slice(PLAYER_CHANNEL);
    INDArray maxPlayerRotation = rotate90(playgroundRotation.slice(MAX_PLAYER_CHANNEL));
    INDArray minPlayerRotation = rotate90(playgroundRotation.slice(MIN_PLAYER_CHANNEL));
    
    INDArray newPlaygroundRotation = createNewBoard(boardEmptyRotation, maxPlayerRotation, minPlayerRotation);

    return newPlaygroundRotation;
  }

  static INDArray createNewBoard(INDArray newEmptyBoardPart, INDArray newMaxPlayerBoardPart, INDArray newMinPlayerBoardPart) {

    INDArray newPlaygroundRotation = Nd4j.create(IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE);
    newPlaygroundRotation.putRow(PLAYER_CHANNEL, newEmptyBoardPart);
    newPlaygroundRotation.putRow(MAX_PLAYER_CHANNEL, newMaxPlayerBoardPart);
    newPlaygroundRotation.putRow(MIN_PLAYER_CHANNEL, newMinPlayerBoardPart);

    return newPlaygroundRotation;
  }
  
  static INDArray rotate90(INDArray toRotate) {
    
    INDArray rotated90 = Nd4j.ones(toRotate.shape()).neg();
    
    int middle = (int) (toRotate.shape()[0] / 2);
    for (int row = 0; row < toRotate.shape()[0]; row++) {
      
      for (int col = 0; col < toRotate.shape()[1]; col++) {
        
        if (row != middle && col != middle) {
          
          if (row == col) {
            
            INDArray slice = toRotate.getColumn(col).dup();
            rotated90.putRow(row, Nd4j.reverse(slice));
          
          } else {
          
            INDArray slice = toRotate.getRow(row).dup();
            rotated90.putColumn(col, slice);
          }
        
        } else if (row == middle && col == middle) {
          
          rotated90.putScalar(new int[]{middle, middle}, toRotate.getDouble(middle, middle));
        }
      }
    }
    
    return rotated90;
  }
}
