package ch.evolutionsoft.rl;

import static org.junit.jupiter.api.Assertions.*;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class MonteCarloTreeSearchTest {

  MonteCarloSearch mcts;
  
  @BeforeEach
  void initMonteCarloTreeSearch() {

    ComputationGraph computationGraph = TestHelper.createConvolutionalConfiguration();
    AdversaryLearningConfiguration adversaryLearningConfiguration =
        new AdversaryLearningConfiguration.Builder().numberOfMonteCarloSimulations(1000).build();
    
    this.mcts = new MonteCarloSearch(computationGraph, adversaryLearningConfiguration);
  }

  @Test
  void checkMonteCarloMoveValidityAndTreeVisitCounts() {
    
    Game game = TestHelper.createMiddlePositionBoardWithThreat();
    
    INDArray actionProbabilities = this.mcts.getActionValues(game, AdversaryLearningConstants.ONE);
    
    TreeNode currentNode = this.mcts.rootNode;
    
    INDArray zeroProbabilityIndices = actionProbabilities.lte(0);

    assertEquals(Nd4j.createFromArray(
        new boolean[] {true, false, true, true, true, false, true, false, false}), zeroProbabilityIndices);
    
    int visitedCountsChildren = 0;
    for (TreeNode rootChildEntry : currentNode.children.values()) {
      
      visitedCountsChildren += rootChildEntry.timesVisited;
    }
    
    int expectedVisitedCountsChildren = 1000 - 1;
    
    assertEquals(expectedVisitedCountsChildren, visitedCountsChildren);
  }
}
