package ch.evolutionsoft.rl4j;

import static ch.evolutionsoft.rl4j.AdversaryLearning.*;
import static ch.evolutionsoft.net.game.tictactoe.TicTacToeConstants.MAX_PLAYER_CHANNEL;
import static ch.evolutionsoft.net.game.tictactoe.TicTacToeConstants.MIN_PLAYER_CHANNEL;

import java.util.ArrayList;
import java.util.Set;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import ch.evolutionsoft.net.game.NeuralNetConstants;
import ch.evolutionsoft.net.game.tictactoe.TicTacToeConstants;
import ch.evolutionsoft.rl4j.tictactoe.PlayoutMain;
import ch.evolutionsoft.rl4j.tictactoe.TicTacToe;

public class AdversaryAgentDriver {

  ComputationGraph player1Policy, player2Policy;
  
  public AdversaryAgentDriver(ComputationGraph player1, ComputationGraph player2) {
    
    this.player1Policy = player1;
    this.player2Policy = player2;
  }

  public int[] playGames(int numberOfEpisodes, INDArray board, double temperature) {
    
    int numberOfEpisodesPlayer1Starts = numberOfEpisodes / 2;
    int numberOfEpisodesPlayer2Starts = numberOfEpisodes - numberOfEpisodesPlayer1Starts;
    
    int player1Wins = 0;
    int player2Wins = 0;
    int draws = 0;
    
    for (int gameNumber = 1; gameNumber <= numberOfEpisodesPlayer1Starts; gameNumber++) {
      
      double gameResult = this.playGame(board, temperature, gameNumber % TicTacToeConstants.COLUMN_COUNT);
      
      if (gameResult >= MAX_WIN) {
        
        player1Wins++;
      
      } else if (gameResult <= MIN_WIN) {
        
        player2Wins++;
      
      } else {
        
        draws++;
      }
    }
    
    ComputationGraph tempPlayerPolicy = player1Policy;
    player1Policy = player2Policy;
    player2Policy = tempPlayerPolicy;

    for (int gameNumber = 1; gameNumber <= numberOfEpisodesPlayer2Starts; gameNumber++) {
      
      double gameResult = this.playGame(board, temperature, gameNumber % TicTacToeConstants.COLUMN_COUNT);
      
      if (gameResult <= MIN_WIN) {
        
        player1Wins++;
      
      } else if (gameResult >= MAX_WIN) {
        
        player2Wins++;
      
      } else {
        
        draws++;
      }
    }
    
    return new int[] {player1Wins, player2Wins, draws};
  }
  
  public double playGame(INDArray board, double temperature, int firstIndex) {
    
    MonteCarloSearch player1 = new MonteCarloSearch(this.player1Policy);
    MonteCarloSearch player2 = new MonteCarloSearch(this.player2Policy);
    
    INDArray currentBoard = PlayoutMain.doFirstMove(firstIndex);
    Set<Integer> emptyFields = TicTacToe.getEmptyFields(currentBoard);
    
    int currentPlayer = TicTacToe.getCurrentPlayer(emptyFields);

    while (!TicTacToe.gameEnded(currentBoard)) {
    
      INDArray moveActionValues = Nd4j.zeros(TicTacToeConstants.COLUMN_COUNT);
      if (currentPlayer == MAX_PLAYER_CHANNEL) {
        
        moveActionValues = player1.getActionValues(currentBoard, temperature);
        
      } else if (currentPlayer == MIN_PLAYER_CHANNEL) {
        
        moveActionValues = player2.getActionValues(currentBoard, temperature);
      }
      
      int moveAction = moveActionValues.argMax(0).getInt(0);

      if (!emptyFields.contains(moveAction)) {
        moveAction = new ArrayList<>(emptyFields).get(NeuralNetConstants.randomGenerator.nextInt(emptyFields.size()));
      }
      
      currentBoard = TicTacToe.makeMove(currentBoard, moveAction, currentPlayer);
      emptyFields = TicTacToe.getEmptyFields(currentBoard);
      currentPlayer = TicTacToe.getCurrentPlayer(emptyFields);
    }
    
    if (TicTacToe.hasWon(currentBoard, MAX_PLAYER_CHANNEL)) {
      
      return MAX_WIN;
    
    }
    
    if (TicTacToe.hasWon(currentBoard, MIN_PLAYER_CHANNEL)) {
      
      return MIN_WIN;
    }
    
    return DRAW_VALUE;
  }
}
