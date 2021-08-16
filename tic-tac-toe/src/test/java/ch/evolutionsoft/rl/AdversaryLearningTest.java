package ch.evolutionsoft.rl;

import static org.junit.jupiter.api.Assertions.*;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import ch.evolutionsoft.rl.tictactoe.ConvolutionResidualNet;
import ch.evolutionsoft.rl.tictactoe.TicTacToe;

public class AdversaryLearningTest {
  
  public static final String TEST_MODEL_BIN = "testModel.bin";
  AdversaryLearning learning;

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
    
    learning = new AdversaryLearning(new TicTacToe(Game.MAX_PLAYER), computationGraph, configuration);
    
    learning.setBestModelName(TEST_MODEL_BIN);
    
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
    
    learning = new AdversaryLearning(new TicTacToe(Game.MAX_PLAYER), computationGraph, configuration);
    
    learning.setBestModelName(TEST_MODEL_BIN);
    
    assertEquals(0, computationGraph.getIterationCount());
    
    learning.performLearning();
    
    assertTrue(1 < computationGraph.getIterationCount());
  }

  @AfterEach
  void deleteTempModel() throws IOException {
    
    Files.delete(Paths.get(learning.getAbsoluteModelPath(TEST_MODEL_BIN)));
  }
  
}
