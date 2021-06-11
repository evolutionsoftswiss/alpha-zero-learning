package ch.evolutionsoft.rl4j.tictactoe;

import static ch.evolutionsoft.net.game.NeuralNetConstants.DOUBLE_COMPARISON_EPSILON;
import static ch.evolutionsoft.net.game.tictactoe.TicTacToeConstants.*;
import static ch.evolutionsoft.net.game.tictactoe.TicTacToeGameHelper.equalsEpsilon;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import ch.evolutionsoft.net.game.tictactoe.TicTacToeConstants;
import ch.evolutionsoft.rl4j.AdversaryTrainingExample;

public class TicTacToe {
  
  public static final int[] COLUMN_INDICES = new int[] {0, 1, 2, 3, 4, 5, 6, 7, 8};
  
  public static List<AdversaryTrainingExample> getSymmetries(INDArray playground, INDArray actionProbabilities, int currentPlayer, int iteration) {
    
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
  
  public static Set<Integer> getEmptyFields(INDArray playground) {
    
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
  
  public static INDArray getValidMoves(INDArray playground) {
    
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
  
  public static int getCurrentPlayer(Set<Integer> emptyFields) {
    
    boolean oddOccupiedFields = 1 == (COLUMN_COUNT - emptyFields.size()) % 2;
    
    if (oddOccupiedFields) {
      
      return MIN_PLAYER_CHANNEL;
    }
    
    return MAX_PLAYER_CHANNEL;
  }
  
  public static int getOtherPlayer(Set<Integer> emptyFields) {
    
    boolean evenOccupiedFields = 0 == (COLUMN_COUNT - emptyFields.size()) % 2;
    
    if (evenOccupiedFields) {
      
      return MIN_PLAYER_CHANNEL;
    }
    
    return MAX_PLAYER_CHANNEL;
  }
  
  public static boolean gameEnded(INDArray board) {

    return TicTacToe.getEmptyFields(board).isEmpty() ||
        TicTacToe.hasWon(board, MAX_PLAYER_CHANNEL) ||
        TicTacToe.hasWon(board, MIN_PLAYER_CHANNEL);
  }

  public static boolean hasWon(INDArray board, int player) {

    return horizontalWin(board, player) || verticalWin(board, player) || diagonalWin(board, player);
  }
  
  public static int getOtherColor(int color) {
    
    if (TicTacToeConstants.MAX_PLAYER_CHANNEL == color) {
      
      return MIN_PLAYER_CHANNEL;
    }
    
    return MAX_PLAYER_CHANNEL;
  }

  public static INDArray makeMove(INDArray board, int flatIndex, int player) {

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
