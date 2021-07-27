package ch.evolutionsoft.rl;

import static ch.evolutionsoft.rl.AdversaryLearning.*;

import java.util.ArrayList;
import java.util.Set;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import ch.evolutionsoft.net.game.NeuralNetConstants;

/**
 * {@link AdversaryAgentDriver} is only relevant if {@link AdversaryLearningConfiguration} has alwaysUpdateNeuralNetwork = false.
 * In that case, a configured number of games and win rate decide if the alpha zero network gets updated with newest version of
 * the neural net, a {@link ComputationGraph} here.
 * 
 * @author evolutionsoft
 */
public class AdversaryAgentDriver {

  ComputationGraph player1Policy, player2Policy;

  Game game;
  
  public AdversaryAgentDriver(Game game, ComputationGraph player1, ComputationGraph player2) {
    
    this.game = game;
    this.player1Policy = player1;
    this.player2Policy = player2;
  }

  public int[] playGames(AdversaryLearningConfiguration configuration, int iteration) {
    
    int numberOfEpisodesPlayer1Starts = configuration.getGamesToGetNewNetworkWinRatio() / 2;
    int numberOfEpisodesPlayer2Starts = configuration.getGamesToGetNewNetworkWinRatio() - numberOfEpisodesPlayer1Starts;
    
    int player1Wins = 0;
    int player2Wins = 0;
    int draws = 0;
    
    for (int gameNumber = 1; gameNumber <= numberOfEpisodesPlayer1Starts; gameNumber++) {
      
      double gameResult = this.playGame(configuration, gameNumber);
      
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
      
      double gameResult = this.playGame(configuration, gameNumber);
      
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
  
  public double playGame(AdversaryLearningConfiguration configuration, int gameNumber) {
    
    MonteCarloSearch player1 = new MonteCarloSearch(this.game, this.player1Policy, configuration);
    MonteCarloSearch player2 = new MonteCarloSearch(this.game, this.player2Policy, configuration);
    
    INDArray currentBoard = game.doFirstMove(gameNumber);
    Set<Integer> emptyFields = game.getValidMoveIndices(currentBoard);
    
    int currentPlayer = Game.MIN_PLAYER;

    while (!game.gameEnded(currentBoard)) {
    
      INDArray moveActionValues = Nd4j.zeros(game.getNumberOfAllAvailableMoves());
      if (currentPlayer == Game.MAX_PLAYER) {
        
        moveActionValues = player1.getActionValues(currentBoard, 0);
        
      } else if (currentPlayer == Game.MIN_PLAYER) {
        
        moveActionValues = player2.getActionValues(currentBoard, 0);
      }
      
      int moveAction = moveActionValues.argMax(0).getInt(0);

      if (!emptyFields.contains(moveAction)) {
        moveAction = new ArrayList<>(emptyFields).get(NeuralNetConstants.randomGenerator.nextInt(emptyFields.size()));
      }
      
      currentBoard = game.makeMove(currentBoard, moveAction, currentPlayer);
      emptyFields = game.getValidMoveIndices(currentBoard);
      currentPlayer = game.getOtherPlayer(currentPlayer);
    }
    
    double endResult = game.getEndResult(currentBoard, currentPlayer);
    if (endResult > 0.5) {

      return MAX_WIN;    

    } else if (endResult < 0.5) {
      
      return MIN_WIN;
    }
    
    return DRAW_VALUE;
  }
}
