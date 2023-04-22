package ch.evolutionsoft.rl.alphazero;

import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import ch.evolutionsoft.rl.AdversaryLearningConfiguration;
import ch.evolutionsoft.rl.Game;

class MonteCarloTreeSearchTest {

  MonteCarloTreeSearch mcts;
  ComputationGraph mockedComputationGraph;
  
  @BeforeEach
  void setupMcts() {
  
    AdversaryLearningConfiguration adversaryLearningConfiguration = new AdversaryLearningConfiguration.Builder().
        numberOfMonteCarloSimulations(10).
        build();
    
    mcts = new MonteCarloTreeSearch(adversaryLearningConfiguration);
    mockedComputationGraph = mock(ComputationGraph.class);
    when(mockedComputationGraph.output(any(INDArray.class))).
      thenReturn(new INDArray[] {Nd4j.ones(2).div(2), Nd4j.ones(1).div(2)});
  }
  
  @Test
  void testMaxSearchMaxWinAfterThree() {
    
    Game mockedGame = new GameMock(0, Game.MAX_WIN, 3, Game.MAX_PLAYER);
    
    TreeNode rootNode = new TreeNode(-2, Game.MAX_PLAYER, 0, 0.5, 0.0, null);
    this.mcts.getActionValues(mockedGame, rootNode, 0, mockedComputationGraph);

    // The mcts value is taken from parent and for parent min here < 0.5
    assertTrue(rootNode.qValue < 0.5);
  }
  
  @Test
  void testMaxSearchMinWinAfterTwo() {
    
    Game mockedGame = new GameMock(0, Game.MIN_WIN, 2, Game.MAX_PLAYER);
    
    TreeNode rootNode = new TreeNode(-2, Game.MAX_PLAYER, 0, 0.5, 0.0, null);
    this.mcts.getActionValues(mockedGame, rootNode, 0, mockedComputationGraph);

    // The mcts value is taken from parent and for parent max here > 0.5
    assertTrue(rootNode.qValue > 0.5);
  }
  
  @Test
  void testMaxSearchMaxWinAfterOne() {
    
    Game mockedGame = new GameMock(0, Game.MAX_WIN, 1, Game.MAX_PLAYER);
    
    TreeNode rootNode = new TreeNode(-2, Game.MAX_PLAYER, 0, 0.5, 0.0, null);
    this.mcts.getActionValues(mockedGame, rootNode, 0, mockedComputationGraph);

    // The mcts value is taken from parent and for parent min here < 0.5
    assertTrue(rootNode.qValue < 0.5);
  }
  
  @Test
  void testMinSearchMaxWinAfterTwo() {
    
    Game mockedGame = new GameMock(0, Game.MAX_WIN, 2, Game.MIN_PLAYER);
    
    TreeNode rootNode = new TreeNode(2, Game.MIN_PLAYER, 0, 0.5, 0.0, null);
    this.mcts.getActionValues(mockedGame, rootNode, 0, mockedComputationGraph);

    // The mcts value is taken from parent and for parent max here > 0.5   
    assertTrue(rootNode.qValue > 0.5);
  }
  
  @Test
  void testMinSearchMinWinAfterOne() {
    
    Game mockedGame = new GameMock(0, Game.MIN_WIN, 1, Game.MIN_PLAYER);
    
    TreeNode rootNode = new TreeNode(2, Game.MIN_PLAYER, 0, 0.5, 0.0, null);
    this.mcts.getActionValues(mockedGame, rootNode, 0, mockedComputationGraph);

    // The mcts value is taken from parent and for parent min here < 0.5   
    assertTrue(rootNode.qValue < 0.5);
  }
}
