package ch.evolutionsoft.rl.alphazero;

import java.util.ArrayList;
import java.util.Set;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import ch.evolutionsoft.rl.AdversaryLearningConfiguration;
import ch.evolutionsoft.rl.AdversaryLearningConstants;
import ch.evolutionsoft.rl.Game;

/**
 * {@link AdversaryAgentDriver} is only relevant if {@link AdversaryLearningConfiguration} has alwaysUpdateNeuralNetwork = false.
 * In that case, a configured number of games and win rate decide if the alpha zero network gets updated with newest version of
 * the neural net, a {@link ComputationGraph} here.
 * 
 * @author evolutionsoft
 */
public class AdversaryAgentDriver {

  ComputationGraph player1Policy;
  ComputationGraph player2Policy;
  
  public AdversaryAgentDriver(ComputationGraph player1, ComputationGraph player2) {
    
    this.player1Policy = player1;
    this.player2Policy = player2;
  }

  public int[] playGames(Game game, AdversaryLearningConfiguration configuration) {
    
    int numberOfEpisodesPlayer1Starts = configuration.getNumberOfGamesToDecideUpdate() / 2;
    int numberOfEpisodesPlayer2Starts = configuration.getNumberOfGamesToDecideUpdate() - numberOfEpisodesPlayer1Starts;
    
    int player1Wins = 0;
    int player2Wins = 0;
    int draws = 0;
    
    for (int gameNumber = 1; gameNumber <= numberOfEpisodesPlayer1Starts; gameNumber++) {
      
      double gameResult = this.playGame(game.createNewInstance(), configuration);
      
      if (gameResult >= Game.MAX_WIN) {
        
        player1Wins++;
      
      } else if (gameResult <= Game.MIN_WIN) {
        
        player2Wins++;
      
      } else {
        
        draws++;
      }
    }
    
    ComputationGraph tempPlayerPolicy = player1Policy;
    player1Policy = player2Policy;
    player2Policy = tempPlayerPolicy;

    for (int gameNumber = 1; gameNumber <= numberOfEpisodesPlayer2Starts; gameNumber++) {
      
      double gameResult = this.playGame(game.createNewInstance(), configuration);
      
      if (gameResult <= Game.MIN_WIN) {
        
        player1Wins++;
      
      } else if (gameResult >= Game.MAX_WIN) {
        
        player2Wins++;
      
      } else {
        
        draws++;
      }
    }
    
    return new int[] {player1Wins, player2Wins, draws};
  }
  
  public double playGame(Game game, AdversaryLearningConfiguration configuration) {
    
    MonteCarloTreeSearch player1 = new MonteCarloTreeSearch(configuration);
    MonteCarloTreeSearch player2 = new MonteCarloTreeSearch(configuration);
    
    Set<Integer> emptyFields = game.getValidMoveIndices();
    
    int currentPlayer = Game.MAX_PLAYER;

    while (!game.gameEnded()) {
    
      INDArray moveActionValues = Nd4j.zeros(game.getNumberOfAllAvailableMoves());
      if (currentPlayer == Game.MAX_PLAYER) {
        
        moveActionValues = player1.getActionValues(game, 0, this.player1Policy);
        
      } else if (currentPlayer == Game.MIN_PLAYER) {
        
        moveActionValues = player2.getActionValues(game, 0, this.player2Policy);
      }
      
      int moveAction = moveActionValues.argMax(0).getInt(0);

      if (!emptyFields.contains(moveAction)) {
        moveAction = new ArrayList<>(emptyFields).get(AdversaryLearningConstants.randomGenerator.nextInt(emptyFields.size()));
      }
      
      game.makeMove(moveAction, currentPlayer);
      emptyFields = game.getValidMoveIndices();
      currentPlayer = game.getOtherPlayer(currentPlayer);
    }
    
    double endResult = game.getEndResult(currentPlayer);
    if (endResult > Game.DRAW) {

      return Game.MAX_WIN;    

    } else if (endResult < Game.DRAW) {
      
      return Game.MIN_WIN;
    }
    
    return Game.DRAW;
  }
}
