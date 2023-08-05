package ch.evolutionsoft.rl.alphazero;

import static org.junit.Assert.*;

import java.io.IOException;
import java.util.List;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

import ch.evolutionsoft.rl.AdversaryLearningConfiguration;
import ch.evolutionsoft.rl.Game;
import ch.evolutionsoft.rl.alphazero.connectfour.ConnectFour;
import ch.evolutionsoft.rl.alphazero.connectfour.ConvolutionResidualNet;
import ch.evolutionsoft.rl.alphazero.connectfour.model.ArrayPosition;
import ch.evolutionsoft.rl.alphazero.connectfour.model.BinaryPlayground;

/**
 * Attention: These tests are sensitiv to network changes and initializations
 * 
 * @author evolutionsoft
 *
 */
class MonteCarloTreeSearchTest {

  private ComputationGraph model;
  private AdversaryLearningConfiguration configuration;
  
  int[] testPosition = new int[]{
      3, 3, 3, 3, 3, 3, 3, 3, 3, 
      3, 0, 1, 0, 0, 1, 1, 2, 3,
      3, 0, 1, 0, 1, 0, 1, 2, 3,
      3, 0, 2, 1, 0, 2, 1, 2, 3,
      3, 2, 2, 0, 1, 2, 0, 2, 3,
      3, 2, 2, 2, 2, 2, 2, 2, 3,
      3, 2, 2, 2, 2, 2, 2, 2, 3,
      3, 3, 3, 3, 3, 3, 3, 3, 3};

  int[] testPosition2 = new int[]{
      3, 3, 3, 3, 3, 3, 3, 3, 3, 
      3, 1, 0, 1, 1, 0, 0, 0, 3,
      3, 1, 0, 1, 0, 1, 0, 2, 3,
      3, 1, 2, 0, 1, 2, 0, 2, 3,
      3, 2, 2, 1, 0, 2, 1, 2, 3,
      3, 2, 2, 2, 2, 2, 2, 2, 3,
      3, 2, 2, 2, 2, 2, 2, 2, 3,
      3, 3, 3, 3, 3, 3, 3, 3, 3};

  int[] testPosition3 = new int[]{
      3, 3, 3, 3, 3, 3, 3, 3, 3, 
      3, 0, 1, 0, 0, 1, 1, 2, 3,
      3, 0, 0, 0, 1, 0, 0, 2, 3,
      3, 1, 1, 1, 0, 2, 0, 2, 3,
      3, 0, 0, 0, 1, 2, 1, 2, 3,
      3, 1, 1, 0, 1, 2, 0, 2, 3,
      3, 0, 1, 1, 1, 2, 2, 2, 3,
      3, 3, 3, 3, 3, 3, 3, 3, 3};

  /*
  * * * * * * * * * 
  * . . . . . . . * 
  * . . O . . . . * 
  * . X O X . . . * 
  * . X O O . . . * 
  * . O X O . X . * 
  * . O X X X O . * 
  * * * * * * * * *
  */
  int[] testPosition4 = new int[]{
      3, 3, 3, 3, 3, 3, 3, 3, 3, 
      3, 2, 1, 1, 0, 0, 1, 2, 3,
      3, 2, 1, 0, 1, 2, 0, 2, 3,
      3, 2, 0, 1, 0, 2, 2, 2, 3,
      3, 2, 0, 1, 0, 2, 2, 2, 3,
      3, 2, 2, 1, 2, 2, 2, 2, 3,
      3, 2, 2, 2, 2, 2, 2, 2, 3,
      3, 3, 3, 3, 3, 3, 3, 3, 3};
  /*
  * * * * * * * * * 
  * . . . X . . . * 
  * . . X O . . . * 
  * O . X O O O . * 
  * X . O X X X . * 
  * X . X O O O O * 
  * X X O X O O X * 
  * * * * * * * * * 
   */
  int[] testPosition5 = new int[]{
      3, 3, 3, 3, 3, 3, 3, 3, 3, 
      3, 0, 0, 1, 0, 1, 1, 2, 3,
      3, 0, 2, 0, 1, 1, 1, 2, 3,
      3, 0, 2, 1, 0, 0, 0, 2, 3,
      3, 1, 2, 0, 1, 1, 1, 2, 3,
      3, 2, 2, 0, 1, 2, 2, 2, 3,
      3, 2, 2, 2, 0, 2, 2, 2, 3,
      3, 3, 3, 3, 3, 3, 3, 3, 3};

  
  @BeforeEach
  void initializeModel() throws IOException {

    configuration = new AdversaryLearningConfiguration.Builder().
        numberOfMonteCarloSimulations(200).
        uctConstantFactor(1.5).
        build();
    ComputationGraphConfiguration modelConfiguration =
        new ConvolutionResidualNet(configuration.getLearningRateSchedule()).createConvolutionalGraphConfiguration();
    model = new ComputationGraph(modelConfiguration);
    model.init();
  }
  
  @Test
  void testMonteCarloSearchSecondPlayerWithThreat() throws IOException {
    
    ArrayPosition arrayPlayground = new ArrayPosition(testPosition, new int[] {3, 2, 4, 4, 2, 4, 0});
    BinaryPlayground binaryPlayground = new BinaryPlayground(arrayPlayground, 23);

    MonteCarloTreeSearch mcts = new MonteCarloTreeSearch(configuration);

    ConnectFour connectFour = new ConnectFour(Game.MIN_PLAYER);
    Game game = connectFour.createNewInstance(binaryPlayground, 0);
    
    INDArray actionValues = mcts.getActionValues(game, 0.5, this.model);
    assertEquals(0, actionValues.argMax(0).getInt(0));
  }
  
  @Test
  void testMonteCarloSearchFirstPlayerWithThreat() throws IOException {
    
    ArrayPosition arrayPlayground = new ArrayPosition(testPosition2, new int[] {3, 2, 4, 4, 2, 4, 1});
    BinaryPlayground binaryPlayground = new BinaryPlayground(arrayPlayground, 22);
    
    MonteCarloTreeSearch mcts = new MonteCarloTreeSearch(configuration);

    ConnectFour connectFour = new ConnectFour();
    Game game = connectFour.createNewInstance(binaryPlayground, 0);
    
    INDArray actionValues = mcts.getActionValues(game, 0.5, this.model);
    assertEquals(4, actionValues.argMax(0).getInt(0));
  }

  @Disabled("Works only with a trained model")
  @Test
  void testMonteCarloSearchFirstPlayerThreatCreationPossible() throws IOException {
    
    ArrayPosition arrayPlayground = new ArrayPosition(testPosition3, new int[] {6, 6, 6, 6, 2, 5, 0});
    BinaryPlayground binaryPlayground = new BinaryPlayground(arrayPlayground, 10);

    MonteCarloTreeSearch mcts = new MonteCarloTreeSearch(configuration);

    ConnectFour connectFour = new ConnectFour();
    Game game = connectFour.createNewInstance(binaryPlayground, 4);
    
    INDArray actionValues = mcts.getActionValues(game, 0.5, this.model);
    assertEquals(4, actionValues.argMax(0).getInt(0));
  }

  @Test
  void testMonteCarloSearchSimpleThreatPosition4() throws IOException {
    
    ArrayPosition arrayPlayground = new ArrayPosition(testPosition4, new int[] {0, 4, 5, 4, 1, 2, 0});
    BinaryPlayground binaryPlayground = new BinaryPlayground(arrayPlayground, 26);
    
    MonteCarloTreeSearch mcts = new MonteCarloTreeSearch(configuration);

    ConnectFour connectFour = new ConnectFour();
    Game game = connectFour.createNewInstance(binaryPlayground, 2);
    
    INDArray actionValues = mcts.getActionValues(game, 0.5, this.model);

    assertTrue(List.of(2, 4).contains(actionValues.argMax(0).getInt(0)));
  }
  
  @Test
  void testMonteCarloSearchSimpleThreatPosition5() throws IOException {
    
    ArrayPosition arrayPlayground = new ArrayPosition(testPosition5, new int[] {4, 1, 5, 6, 4, 4, 0});
    BinaryPlayground binaryPlayground = new BinaryPlayground(arrayPlayground, 17);
    
    MonteCarloTreeSearch mcts = new MonteCarloTreeSearch(configuration);

    ConnectFour connectFour = new ConnectFour();
    Game game = connectFour.createNewInstance(binaryPlayground, 2);
    
    INDArray actionValues = mcts.getActionValues(game, 1.0, this.model);
    assertNotEquals(6, actionValues.argMax(0).getInt(0));
  }
}
