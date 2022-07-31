package ch.evolutionsoft.rl.alphazero;

import static org.junit.Assert.*;

import java.io.IOException;

import org.deeplearning4j.nn.conf.CNN2DFormat;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

import ch.evolutionsoft.rl.AdversaryLearningConfiguration;
import ch.evolutionsoft.rl.Game;
import ch.evolutionsoft.rl.alphazero.connectfour.ConnectFour;
import ch.evolutionsoft.rl.alphazero.connectfour.ConvolutionResidualNet;
import ch.evolutionsoft.rl.alphazero.connectfour.playground.ArrayPlayground;

class MonteCarloTreeSearchTest {

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
      3, 1, 0, 1, 1, 0, 0, 1, 3,
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
  
  @Test
  void testMonteCarloSearchSecondPlayerWithThreat() throws IOException {
    
    ArrayPlayground arrayPlayground = new ArrayPlayground(testPosition, new int[] {3, 2, 4, 4, 2, 4, 0});
    AdversaryLearningConfiguration configuration = new AdversaryLearningConfiguration.Builder().
        numberOfMonteCarloSimulations(300).
        uctConstantFactor(1.5).
        build();

    ComputationGraphConfiguration modelConfiguration =
        new ConvolutionResidualNet(configuration.getLearningRateSchedule()).createConvolutionalGraphConfiguration();
    ComputationGraph model = new ComputationGraph(modelConfiguration);
    model.init();

    MonteCarloTreeSearch mcts = new MonteCarloTreeSearch(configuration);

    ConnectFour connectFour = new ConnectFour();
    Game game = connectFour.createNewInstance(arrayPlayground);
    
    INDArray actionValues = mcts.getActionValues(game, 0.5, model);
    assertEquals(0, actionValues.argMax(0).getInt(0));
  }
  
  @Test
  void testMonteCarloSearchFirstPlayerWithThreat() throws IOException {
    
    ArrayPlayground arrayPlayground = new ArrayPlayground(testPosition2, new int[] {3, 2, 4, 4, 2, 4, 1});
    AdversaryLearningConfiguration configuration = new AdversaryLearningConfiguration.Builder().
        numberOfMonteCarloSimulations(300).
        uctConstantFactor(1.5).
        build();

    ComputationGraphConfiguration modelConfiguration =
        new ConvolutionResidualNet(configuration.getLearningRateSchedule()).createConvolutionalGraphConfiguration();
    ComputationGraph model = new ComputationGraph(modelConfiguration);
    model.init();
    
    MonteCarloTreeSearch mcts = new MonteCarloTreeSearch(configuration);

    ConnectFour connectFour = new ConnectFour();
    Game game = connectFour.createNewInstance(arrayPlayground);
    
    INDArray actionValues = mcts.getActionValues(game, 0.5, model);
    assertEquals(0, actionValues.argMax(0).getInt(0));
  }
  
  @Disabled
  @Test
  void testMonteCarloSearchSecondPlayer3() throws IOException {
    
    ArrayPlayground arrayPlayground = new ArrayPlayground(testPosition3, new int[] {6, 6, 6, 6, 3, 5, 0});
    AdversaryLearningConfiguration configuration = new AdversaryLearningConfiguration.Builder().
        numberOfMonteCarloSimulations(300).
        uctConstantFactor(1.5).
        build();

    ComputationGraphConfiguration modelConfiguration =
        new ConvolutionResidualNet(configuration.getLearningRateSchedule()).createConvolutionalGraphConfiguration();
    ComputationGraph model = new ComputationGraph(modelConfiguration);
    model.init();
    
    MonteCarloTreeSearch mcts = new MonteCarloTreeSearch(configuration);

    ConnectFour connectFour = new ConnectFour();
    Game game = connectFour.createNewInstance(arrayPlayground);
    
    INDArray actionValues = mcts.getActionValues(game, 0.5, model);
    assertNotEquals(4, actionValues.argMax(0).getInt(0));
  }
}
