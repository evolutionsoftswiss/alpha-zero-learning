package ch.evolutionsoft.rl.alphazero;

import static org.junit.jupiter.api.Assertions.*;

import java.io.IOException;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import ch.evolutionsoft.rl.AdversaryLearningConfiguration;
import ch.evolutionsoft.rl.Game;

class MonteCarloTreeSearchTest {
  
  ComputationGraph computationGraph;
  
  @BeforeEach
  void initMonteCarloTreeSearch() {

    computationGraph = TestHelper.createConvolutionalConfiguration();
  }

  @Test
  void checkMonteCarloMoveValidityAndTreeVisitCounts() throws IOException {
    
    Game game = TestHelper.createMiddlePositionBoardWithThreat();
    AdversaryLearningConfiguration adversaryLearningConfiguration =
        new AdversaryLearningConfiguration.Builder().
        numberOfEpisodesBeforePotentialUpdate(10).
        numberOfEpisodeThreads(16).
        numberOfMonteCarloSimulations(1000).
        build();

    TreeNode rootNode = new TreeNode(-1, game.getOtherPlayer(game.getCurrentPlayer()), 0, 1.0, 0.5, null);
    MonteCarloTreeSearch mcts = new MonteCarloTreeSearch(adversaryLearningConfiguration);

    INDArray actionProbabilities = mcts.getActionValues(game, rootNode, 0, computationGraph);
    
    INDArray zeroProbabilityIndices = actionProbabilities.lte(0);

    assertEquals(Nd4j.createFromArray(
        new boolean[] {true, false, true, true, true, true, true, true, true}), zeroProbabilityIndices);
    
    int visitedCountsChildren = 0;
    for (TreeNode rootChildEntry : rootNode.children.values()) {
      
      visitedCountsChildren += rootChildEntry.timesVisited;
    }
    
    int expectedVisitedCountsChildren = 1000 - 1;
    
    assertEquals(expectedVisitedCountsChildren, visitedCountsChildren);
  }

  @Test
  void checkMonteCarloMoveSelect() throws IOException {
    
    Game game = TestHelper.createMiddlePositionBoardWithThreat();
    AdversaryLearningConfiguration adversaryLearningConfiguration =
        new AdversaryLearningConfiguration.Builder().
        numberOfEpisodesBeforePotentialUpdate(10).
        numberOfMonteCarloSimulations(1000).
        build();

    MonteCarloTreeSearch mcts = new MonteCarloTreeSearch(adversaryLearningConfiguration);
    
    INDArray actionProbabilities =
        mcts.getActionValues(game, 1.0, computationGraph);
    
    assertEquals(1, actionProbabilities.argMax(0).getInt(0));
  }
}
