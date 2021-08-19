package ch.evolutionsoft.rl.tictactoe;

import java.io.IOException;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ch.evolutionsoft.net.game.tictactoe.TicTacToeConstants;
import ch.evolutionsoft.rl.AdversaryLearning;
import ch.evolutionsoft.rl.AdversaryLearningConfiguration;
import ch.evolutionsoft.rl.Game;
import ch.evolutionsoft.rl.MonteCarloSearch;

public class TicTacToeGamesMain {

  private static final Logger log = LoggerFactory.getLogger(TicTacToeGamesMain.class);

  public static void main(String[] args) throws IOException {
    
    AdversaryLearningConfiguration configuration = new AdversaryLearningConfiguration.Builder().build();
    
    ComputationGraph perfectResNet = ModelSerializer.restoreComputationGraph(
        configuration.getAbsoluteModelPathFrom("TicTacToePerfectResidualNet.bin"));
    ComputationGraph alphaNet = ModelSerializer.restoreComputationGraph(
        configuration.getAbsoluteModelPathFrom("bestmodel.bin"));
    
    int[] results1 = playGamesSupervisedNetVsAlphaZeroNet(perfectResNet, alphaNet);
    int[] results2 = playGamesAlphaNetVsSupervisedResidualNet(perfectResNet, alphaNet);

    log.info("Alpha O: loss {} draws {} wins {}", results1[0], results1[1], results1[2]);
    log.info("Alpha X: loss {} draws {} wins {}", results2[2], results2[1], results2[0]);
  }

  static int[] playGamesSupervisedNetVsAlphaZeroNet(ComputationGraph perfectResNet, ComputationGraph alphaNet) {

    int[] results = new int[3];
    
    for (int game = 1; game <= 27; game++) {

      Game ticTacToe = new TicTacToe(Game.MAX_PLAYER);

      int firstMoveIndex = game % TicTacToeConstants.COLUMN_COUNT;
      ticTacToe.doFirstMove(firstMoveIndex);
      boolean xPlayer = false;
      int numberOfMoves = 1;
      
      while (!ticTacToe.gameEnded()) {
        
        if (xPlayer) {

          doMoveFromSupervisedResidualNet(perfectResNet, ticTacToe, xPlayer);
        
        } else {

          doMoveFromAlphaZeroNet(alphaNet, ticTacToe, xPlayer);
        }
        
        numberOfMoves++;
        xPlayer = !xPlayer;
        
      }

      double endResult = ticTacToe.getEndResult(-1);
      
      if (endResult > AdversaryLearning.DRAW_VALUE) {

        log.info("X wins after {} moves", numberOfMoves);
        results[0]++;
      
      } else if (endResult < AdversaryLearning.DRAW_VALUE) {

        log.info("O wins after {} moves", numberOfMoves);
        results[2]++;
      
      } else if (ticTacToe.getValidMoveIndices().isEmpty()) {

        log.info("Draw");
        results[1]++;
      }

      log.info("Game finished with first move index {}\nGame ended with board {}", firstMoveIndex, ticTacToe.getCurrentBoard());
    }
    
    return results;
  }

  static int[] playGamesAlphaNetVsSupervisedResidualNet(ComputationGraph perfectResNet, ComputationGraph alphaNet) {

    int[] results = new int[3];
    
    for (int game = 1; game <= 27; game++) {

      Game ticTacToe = new TicTacToe(Game.MAX_PLAYER);
      
      int firstMoveIndex = game % TicTacToeConstants.COLUMN_COUNT;
      ticTacToe.doFirstMove(firstMoveIndex);
      boolean xPlayer = false;
      int numberOfMoves = 1;
      
      while (!ticTacToe.gameEnded()) {
        
        if (!xPlayer) {

          doMoveFromSupervisedResidualNet(perfectResNet, ticTacToe, xPlayer);
        
        } else {

          doMoveFromAlphaZeroNet(alphaNet, ticTacToe, xPlayer);
        }
        
        numberOfMoves++;
        xPlayer = !xPlayer;
        
      }

      double endResult = ticTacToe.getEndResult(-1);      
      if (endResult > AdversaryLearning.DRAW_VALUE) {

        log.info("X wins after {} moves", numberOfMoves);
        results[0]++;
      
      } else if (endResult < AdversaryLearning.DRAW_VALUE) {

        log.info("O wins after {} moves", numberOfMoves);
        results[2]++;
      
      } else if (ticTacToe.getValidMoveIndices().isEmpty()) {

        log.info("Draw");
        results[1]++;
        
      }

      log.info("Game finished with first move index {}\nGame ended with board {}", firstMoveIndex, ticTacToe.getCurrentBoard());
    }
    
    return results;
  }

  static INDArray doMoveFromAlphaZeroNet(ComputationGraph alphaNet, Game ticTacToe, boolean xPlayer) {
    int moveIndex = new MonteCarloSearch(alphaNet, new AdversaryLearningConfiguration.
        Builder().build()).getActionValues(ticTacToe, 0).argMax(0).getInt(0);
    
    if (!ticTacToe.getValidMoveIndices().contains(moveIndex)) {
      log.warn("Invalid O move from alpha zero net.");
      moveIndex = ticTacToe.getValidMoveIndices().iterator().next();
    }
    
    return ticTacToe.makeMove(moveIndex, xPlayer ? TicTacToeConstants.MAX_PLAYER_CHANNEL : TicTacToeConstants.MIN_PLAYER_CHANNEL);
  }

  static INDArray doMoveFromSupervisedResidualNet(ComputationGraph perfectResNet, Game ticTacToe,
      boolean xPlayer) {
    int moveIndex = getBestMove(perfectResNet, ticTacToe.getCurrentBoard());

    if (!ticTacToe.getValidMoveIndices().contains(moveIndex)) {
      log.warn("Invalid O move from potentially perfect residual net.");
      moveIndex = ticTacToe.getValidMoveIndices().iterator().next();
    }
    
    return ticTacToe.makeMove(moveIndex, xPlayer ? TicTacToeConstants.MAX_PLAYER_CHANNEL : TicTacToeConstants.MIN_PLAYER_CHANNEL);
  }

  public static int getBestMove(ComputationGraph computationGraph, INDArray board) {

    INDArray inputBoardBatch = Nd4j.zeros(1, TicTacToeConstants.IMAGE_CHANNELS, TicTacToeConstants.IMAGE_SIZE, TicTacToeConstants.IMAGE_SIZE);
    inputBoardBatch.putRow(0, board);
    INDArray[] output = computationGraph.output(inputBoardBatch);

    return output[0].argMax(1).getInt(0);
  }
  
}
