package ch.evolutionsoft.rl4j;

import static ch.evolutionsoft.net.game.tictactoe.TicTacToeConstants.MAX_PLAYER_CHANNEL;
import static ch.evolutionsoft.net.game.tictactoe.TicTacToeConstants.MIN_PLAYER_CHANNEL;

import java.util.Set;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import ch.evolutionsoft.net.game.NeuralNetConstants;
import ch.evolutionsoft.net.game.tictactoe.TicTacToeConstants;
import ch.evolutionsoft.rl4j.tictactoe.TicTacToe;

public class AdversaryAgentDriver {

  MonteCarloTreeSearch player1, player2;
  
  public AdversaryAgentDriver(MonteCarloTreeSearch player1, MonteCarloTreeSearch player2) {
    
    this.player1 = player1;
    this.player2 = player2;
  }

  public int[] playGames(int numberOfEpisodes, INDArray board) {
    
    int numberOfEpisodesPlayer1Starts = numberOfEpisodes / 2;
    int numberOfEpisodesPlayer2Starts = numberOfEpisodes - numberOfEpisodesPlayer1Starts;
    
    int player1Wins = 0;
    int player2Wins = 0;
    int draws = 0;
    
    for (int gameNumber = 1; gameNumber <= numberOfEpisodesPlayer1Starts; gameNumber++) {
      
      float gameResult = this.playGame(board);
      
      if (gameResult >= 1) {
        
        player1Wins++;
      
      } else if (gameResult <= -1) {
        
        player2Wins++;
      
      } else {
        
        draws++;
      }
    }
    
    MonteCarloTreeSearch tempPlayer = player1;
    player1 = player2;
    player2 = tempPlayer;

    for (int gameNumber = 1; gameNumber <= numberOfEpisodesPlayer2Starts; gameNumber++) {
      
      float gameResult = this.playGame(board);
      
      if (gameResult <= -1) {
        
        player1Wins++;
      
      } else if (gameResult >= 1) {
        
        player2Wins++;
      
      } else {
        
        draws++;
      }
    }
    
    return new int[] {player1Wins, player2Wins, draws};
  }
  
  public float playGame(INDArray board) {

    INDArray currentBoard = board.dup();
    
    Set<Integer> emptyFields = TicTacToe.getEmptyFields(board);
    int currentPlayer = TicTacToe.getCurrentPlayer(emptyFields);
    while (!TicTacToe.gameEnded(currentBoard)) {
    
      INDArray moveActionValues = Nd4j.zeros(TicTacToeConstants.COLUMN_COUNT);
      if (currentPlayer == MAX_PLAYER_CHANNEL) {
        
        moveActionValues = player1.getActionValues(currentBoard, 0);
        
      } else if (currentPlayer == MIN_PLAYER_CHANNEL) {
        
        moveActionValues = player2.getActionValues(currentBoard, 0);
      }
      
      INDArray maxMoveValueIndices = moveActionValues.argMax(0);
      
      for (int maxMoveValueIndex = 0; maxMoveValueIndex < maxMoveValueIndices.size(0); maxMoveValueIndex++) {
        
        if (!emptyFields.contains(maxMoveValueIndices.getInt(maxMoveValueIndex))) {
          
          throw new IllegalStateException("Move " + maxMoveValueIndices.getInt(maxMoveValueIndex) +
              " is not valid on board " + currentBoard);
        }
      }
      
      int moveAction = maxMoveValueIndices.getInt(
          NeuralNetConstants.randomGenerator.nextInt((int) maxMoveValueIndices.size(0)));
      
      currentBoard = TicTacToe.makeMove(currentBoard, moveAction, currentPlayer);
      emptyFields = TicTacToe.getEmptyFields(currentBoard);
      currentPlayer = TicTacToe.getCurrentPlayer(emptyFields);
    }
    
    if (TicTacToe.hasWon(currentBoard, MAX_PLAYER_CHANNEL)) {
      
      return 1;
    
    }
    
    if (TicTacToe.hasWon(currentBoard, MIN_PLAYER_CHANNEL)) {
      
      return -1;
    }
    
    return 0;
  }
}
