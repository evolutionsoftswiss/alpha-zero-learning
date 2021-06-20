package ch.evolutionsoft.rl.tictactoe;

import java.io.IOException;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import ch.evolutionsoft.net.game.tictactoe.TicTacToeConstants;
import ch.evolutionsoft.rl.AdversaryLearningConfiguration;
import ch.evolutionsoft.rl.Game;
import ch.evolutionsoft.rl.MonteCarloSearch;

public class TicTacToePlayoutMain {

  public static void main(String[] args) throws IOException {
    
    ComputationGraph perfectResNet = ModelSerializer.restoreComputationGraph("TicTacToePerfectResidualNet.bin");
    ComputationGraph alphaNet = ModelSerializer.restoreComputationGraph("bestmodel.bin");
    
    int draws1 = 0;
    int xWins1 = 0;
    int oWins1 = 0;

    Game ticTacToe = new TicTacToe();
    
    for (int game = 1; game <= 27; game++) {

      boolean xPlayer = false;
      INDArray board = ticTacToe.doFirstMove(game % TicTacToeConstants.COLUMN_COUNT);
      int numberOfMoves = 1;
      
      while (!ticTacToe.gameEnded(board)) {
        
        if (xPlayer) {

          int moveIndex = getBestMove(perfectResNet, board);
          
          if (!ticTacToe.getEmptyFields(board).contains(moveIndex)) {
            System.out.println("Invalid X move");
            moveIndex = ticTacToe.getEmptyFields(board).iterator().next();
          }
          
          board = ticTacToe.makeMove(board, moveIndex, TicTacToeConstants.MAX_PLAYER_CHANNEL);
        
        } else {

          int moveIndex = new MonteCarloSearch(ticTacToe, alphaNet, new AdversaryLearningConfiguration.Builder().build(), board).getActionValues(board, 1).argMax(0).getInt(0);
          
          if (!ticTacToe.getEmptyFields(board).contains(moveIndex)) {
            System.out.println("Invalid O move");
            moveIndex = ticTacToe.getEmptyFields(board).iterator().next();
          }
          
          board = ticTacToe.makeMove(board, moveIndex, TicTacToeConstants.MIN_PLAYER_CHANNEL);
        }
        
        numberOfMoves++;
        xPlayer = !xPlayer;
        
      }
      
      if (ticTacToe.hasWon(board, TicTacToeConstants.MAX_PLAYER_CHANNEL)) {

        System.out.println("X wins after " + numberOfMoves + " moves");
        xWins1++;
      
      } else if (ticTacToe.hasWon(board, TicTacToeConstants.MIN_PLAYER_CHANNEL)) {

        System.out.println("O wins after " + numberOfMoves + " moves");
        oWins1++;
      
      } else if (ticTacToe.getEmptyFields(board).isEmpty()) {

        System.out.println("Draw");
        draws1++;
        
      } else {
        
        System.out.println("Error");
      }
     
      numberOfMoves = 1;
      System.out.println(board);
    }
    
    int draws2 = 0;
    int xWins2 = 0;
    int oWins2 = 0;
    
    for (int game = 1; game <= 27; game++) {
      
      boolean xPlayer = false;
      INDArray board = ticTacToe.doFirstMove(game % TicTacToeConstants.COLUMN_COUNT);
      int numberOfMoves = 1;
      
      while (!ticTacToe.gameEnded(board)) {
        
        if (!xPlayer) {

          int moveIndex = getBestMove(perfectResNet, board);

          if (!ticTacToe.getEmptyFields(board).contains(moveIndex)) {
            System.out.println("Invalid O move");
            moveIndex = ticTacToe.getEmptyFields(board).iterator().next();
          }
          
          board = ticTacToe.makeMove(board, moveIndex, TicTacToeConstants.MIN_PLAYER_CHANNEL);
        
        } else {

          int moveIndex = new MonteCarloSearch(ticTacToe, alphaNet, new AdversaryLearningConfiguration.Builder().build(), board).getActionValues(board, 1).argMax(0).getInt(0);
          
          if (!ticTacToe.getEmptyFields(board).contains(moveIndex)) {
            System.out.println("Invalid X move");
            moveIndex = ticTacToe.getEmptyFields(board).iterator().next();
          }
          
          board = ticTacToe.makeMove(board, moveIndex, TicTacToeConstants.MAX_PLAYER_CHANNEL);
        }
        
        numberOfMoves++;
        xPlayer = !xPlayer;
        
      }
      
      if (ticTacToe.hasWon(board, TicTacToeConstants.MAX_PLAYER_CHANNEL)) {

        System.out.println("X wins after " + numberOfMoves + " moves");
        xWins2++;
      
      } else if (ticTacToe.hasWon(board, TicTacToeConstants.MIN_PLAYER_CHANNEL)) {

        System.out.println("O wins after " + numberOfMoves + " moves");
        oWins2++;
      
      } else if (ticTacToe.getEmptyFields(board).isEmpty()) {

        System.out.println("Draw");
        draws2++;
        
      } else {
        
        System.out.println("Error");
      }
     
      numberOfMoves = 0;
      System.out.println(board);
    }

    System.out.println("Alpha O: loss " + xWins1 + " draws " + draws1 + " wins " + oWins1);
    System.out.println("Alpha X: loss " + oWins2 + " draws " + draws2 + " wins " + xWins2);
    
  }

  public static int getBestMove(ComputationGraph computationGraph, INDArray board) {

    INDArray inputBoardBatch = Nd4j.zeros(1, 3, 3, 3);
    inputBoardBatch.putRow(0, board);
    INDArray[] output = computationGraph.output(inputBoardBatch);
    int moveIndex = output[0].argMax(1).getInt(0);
    return moveIndex;
  }
  
}
