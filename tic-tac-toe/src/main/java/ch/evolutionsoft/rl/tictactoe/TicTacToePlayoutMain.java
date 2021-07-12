package ch.evolutionsoft.rl.tictactoe;

import java.io.IOException;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ch.evolutionsoft.net.game.tictactoe.TicTacToeConstants;
import ch.evolutionsoft.rl.AdversaryLearningConfiguration;
import ch.evolutionsoft.rl.Game;
import ch.evolutionsoft.rl.MonteCarloSearch;

public class TicTacToePlayoutMain {

  private static final Logger log = LoggerFactory.getLogger(TicTacToePlayoutMain.class);

  public static void main(String[] args) throws IOException {
    
    ComputationGraph perfectResNet = ModelSerializer.restoreComputationGraph("TicTacToePerfectResidualNet.bin");
    ComputationGraph alphaNet = ModelSerializer.restoreComputationGraph("bestmodel.bin");
    
    playGamesSupervisedNetVsAlphaZeroNet(perfectResNet, alphaNet);
    playGamesAlphaNetVsSupervisedResidualNet(perfectResNet, alphaNet);

  }

  static void playGamesSupervisedNetVsAlphaZeroNet(ComputationGraph perfectResNet, ComputationGraph alphaNet) {

    int draws1 = 0;
    int xWins1 = 0;
    int oWins1 = 0;
    
    for (int game = 1; game <= 27; game++) {

      Game ticTacToe = new TicTacToe(Game.MAX_PLAYER);

      int firstMoveIndex = game % TicTacToeConstants.COLUMN_COUNT;
      INDArray board = ticTacToe.doFirstMove(firstMoveIndex);
      boolean xPlayer = false;
      int numberOfMoves = 1;
      
      while (!ticTacToe.gameEnded(board)) {
        
        if (xPlayer) {

          board = doMoveFromSupervisedResidualNet(perfectResNet, ticTacToe, board, xPlayer);
        
        } else {

          board = doMoveFromAlphaZeroNet(alphaNet, ticTacToe, board, xPlayer);
        }
        
        numberOfMoves++;
        xPlayer = !xPlayer;
        
      }
      
      if (ticTacToe.hasWon(board, TicTacToeConstants.MAX_PLAYER_CHANNEL)) {

        log.info("X wins after {} moves", numberOfMoves);
        xWins1++;
      
      } else if (ticTacToe.hasWon(board, TicTacToeConstants.MIN_PLAYER_CHANNEL)) {

        log.info("O wins after {} moves", numberOfMoves);
        oWins1++;
      
      } else if (ticTacToe.getValidMoveIndices(board).isEmpty()) {

        log.info("Draw");
        draws1++;
        
      }

      log.info("Playout finished with first move index {}\nGame ended with board {}", firstMoveIndex, board);
    }
    log.info("Alpha O: loss {} draws {} wins {}", xWins1, draws1, oWins1);
  }

  static void playGamesAlphaNetVsSupervisedResidualNet(ComputationGraph perfectResNet, ComputationGraph alphaNet) {

    int draws2 = 0;
    int xWins2 = 0;
    int oWins2 = 0;
    
    for (int game = 1; game <= 27; game++) {

      Game ticTacToe = new TicTacToe(Game.MAX_PLAYER);
      
      int firstMoveIndex = game % TicTacToeConstants.COLUMN_COUNT;
      INDArray board = ticTacToe.doFirstMove(firstMoveIndex);
      boolean xPlayer = false;
      int numberOfMoves = 1;
      
      while (!ticTacToe.gameEnded(board)) {
        
        if (!xPlayer) {

          board = doMoveFromSupervisedResidualNet(perfectResNet, ticTacToe, board, xPlayer);
        
        } else {

          board = doMoveFromAlphaZeroNet(alphaNet, ticTacToe, board, xPlayer);
        }
        
        numberOfMoves++;
        xPlayer = !xPlayer;
        
      }
      
      if (ticTacToe.hasWon(board, TicTacToeConstants.MAX_PLAYER_CHANNEL)) {

        log.info("X wins after {} moves", numberOfMoves);
        xWins2++;
      
      } else if (ticTacToe.hasWon(board, TicTacToeConstants.MIN_PLAYER_CHANNEL)) {

        log.info("O wins after {} moves", numberOfMoves);
        oWins2++;
      
      } else if (ticTacToe.getValidMoveIndices(board).isEmpty()) {

        log.info("Draw");
        draws2++;
        
      }

      log.info("Playout finished with first move index {}\nGame ended with board {}", firstMoveIndex, board);
    }
    log.info("Alpha X: loss {} draws {} wins {}", oWins2, draws2, xWins2);
  }

  static INDArray doMoveFromAlphaZeroNet(ComputationGraph alphaNet, Game ticTacToe, INDArray board, boolean xPlayer) {
    int moveIndex = new MonteCarloSearch(ticTacToe, alphaNet, new AdversaryLearningConfiguration.
        Builder().build(), board).getActionValues(board, 0).argMax(0).getInt(0);
    
    if (!ticTacToe.getValidMoveIndices(board).contains(moveIndex)) {
      log.warn("Invalid O move from alpha zero net.");
      moveIndex = ticTacToe.getValidMoveIndices(board).iterator().next();
    }
    
    board = ticTacToe.makeMove(board, moveIndex, xPlayer ? TicTacToeConstants.MAX_PLAYER_CHANNEL : TicTacToeConstants.MIN_PLAYER_CHANNEL);
    return board;
  }

  static INDArray doMoveFromSupervisedResidualNet(ComputationGraph perfectResNet, Game ticTacToe, INDArray board,
      boolean xPlayer) {
    int moveIndex = getBestMove(perfectResNet, board);

    if (!ticTacToe.getValidMoveIndices(board).contains(moveIndex)) {
      log.warn("Invalid O move from potentially perfect residual net.");
      moveIndex = ticTacToe.getValidMoveIndices(board).iterator().next();
    }
    
    board = ticTacToe.makeMove(board, moveIndex, xPlayer ? TicTacToeConstants.MAX_PLAYER_CHANNEL : TicTacToeConstants.MIN_PLAYER_CHANNEL);
    return board;
  }

  public static int getBestMove(ComputationGraph computationGraph, INDArray board) {

    INDArray inputBoardBatch = Nd4j.zeros(1, TicTacToeConstants.IMAGE_CHANNELS, TicTacToeConstants.IMAGE_SIZE, TicTacToeConstants.IMAGE_SIZE);
    inputBoardBatch.putRow(0, board);
    INDArray[] output = computationGraph.output(inputBoardBatch);

    return output[0].argMax(1).getInt(0);
  }
  
}
