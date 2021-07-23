package ch.evolutionsoft.rl;

import static org.junit.jupiter.api.Assertions.*;

import java.io.IOException;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.junit.jupiter.api.Test;

import ch.evolutionsoft.rl.tictactoe.ConvolutionResidualNet;
import ch.evolutionsoft.rl.tictactoe.TicTacToe;

public class AdversaryLearningTest {

  @Test
  void testNetUpdateLastIteration() throws IOException {
    
    AdversaryLearningConfiguration configuration =
        new AdversaryLearningConfiguration.Builder().
        alwaysUpdateNeuralNetwork(true).
        numberOfIterations(1).
        numberOfIterationsBeforePotentialUpdate(1).
        build();
 
    ComputationGraph computationGraph =
        new ComputationGraph(new ConvolutionResidualNet().createConvolutionalGraphConfiguration());
    computationGraph.init();
    
    AdversaryLearning learning = new AdversaryLearning(new TicTacToe(Game.MAX_PLAYER), computationGraph, configuration);
    
    assertEquals(0, computationGraph.getIterationCount());
    
    learning.performLearning();
    
    assertEquals(1, computationGraph.getIterationCount());
  }

  @Test
  void testMiniBatchBehavior() throws IOException {
    
    AdversaryLearningConfiguration configuration =
        new AdversaryLearningConfiguration.Builder().
        alwaysUpdateNeuralNetwork(true).
        numberOfIterations(1).
        numberOfIterationsBeforePotentialUpdate(1).
        batchSize(16).
        build();
 
    ComputationGraph computationGraph =
        new ComputationGraph(new ConvolutionResidualNet().createConvolutionalGraphConfiguration());
    computationGraph.init();
    
    AdversaryLearning learning = new AdversaryLearning(new TicTacToe(Game.MAX_PLAYER), computationGraph, configuration);
    
    assertEquals(0, computationGraph.getIterationCount());
    
    learning.performLearning();
    
    assertTrue(1 < computationGraph.getIterationCount());
  }

}
