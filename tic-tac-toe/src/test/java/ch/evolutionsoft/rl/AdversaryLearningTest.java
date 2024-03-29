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
  public static final String TEST_TRAIN_EXAMPLES = "testTrainExamples.obj";
  
  AdversaryLearningConfiguration configuration;

  @Test
  void testNetUpdateLastIteration() throws IOException {

    configuration =
        new AdversaryLearningConfiguration.Builder().
        alwaysUpdateNeuralNetwork(true).
        numberOfIterations(1).
        numberOfIterationsBeforePotentialUpdate(1).
        bestModelFileName(TEST_MODEL_BIN).
        trainExamplesFileName(TEST_TRAIN_EXAMPLES).
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
    
    configuration =
        new AdversaryLearningConfiguration.Builder().
        alwaysUpdateNeuralNetwork(true).
        numberOfIterations(1).
        numberOfIterationsBeforePotentialUpdate(1).
        batchSize(16).
        bestModelFileName(TEST_MODEL_BIN).
        trainExamplesFileName(TEST_TRAIN_EXAMPLES).
        build();
 
    ComputationGraph computationGraph =
        new ComputationGraph(new ConvolutionResidualNet().createConvolutionalGraphConfiguration());
    computationGraph.init();
    
    AdversaryLearning learning = new AdversaryLearning(new TicTacToe(Game.MAX_PLAYER), computationGraph, configuration);
    
    assertEquals(0, computationGraph.getIterationCount());
    
    learning.performLearning();
    
    assertTrue(1 < computationGraph.getIterationCount());
  }

  @Test
  void testChallengeGames() throws IOException {
    
    configuration =
        new AdversaryLearningConfiguration.Builder().
        alwaysUpdateNeuralNetwork(false).
        numberOfGamesToDecideUpdate(1).
        gamesWinRatioThresholdNewNetworkUpdate(-0.1).
        numberOfIterations(1).
        numberOfIterationsBeforePotentialUpdate(1).
        bestModelFileName(TEST_MODEL_BIN).
        trainExamplesFileName(TEST_TRAIN_EXAMPLES).
        build();
 
    ComputationGraph computationGraph =
        new ComputationGraph(new ConvolutionResidualNet().createConvolutionalGraphConfiguration());
    computationGraph.init();
    
    AdversaryLearning learning = new AdversaryLearning(new TicTacToe(Game.MAX_PLAYER), computationGraph, configuration);
    
    assertEquals(0, computationGraph.getIterationCount());
    
    learning.performLearning();
    
    assertEquals(1, computationGraph.getIterationCount());

    Files.delete(Paths.get(configuration.getAbsoluteModelPathFrom("tempmodel.bin")));
  }

  @AfterEach
  void deleteTempModel() throws IOException {

    Files.delete(Paths.get(configuration.getAbsoluteModelPathFrom(TEST_MODEL_BIN)));
    Files.delete(Paths.get(configuration.getAbsoluteModelPathFrom(TEST_TRAIN_EXAMPLES)));
  }
  
}
