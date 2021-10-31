package ch.evolutionsoft.rl;

import static org.junit.jupiter.api.Assertions.*;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.TimeUnit;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import ch.evolutionsoft.rl.tictactoe.ConvolutionResidualNet;
import ch.evolutionsoft.rl.tictactoe.TicTacToe;

class AdversaryLearningTicTacToeTest {
  
  public static final String TEST_MODEL_BIN = "testModel.bin";
  public static final String TEST_TRAIN_EXAMPLES = "testTrainExamples.obj";
  public static final String TEST_TRAIN_EXAMPLES_VALUES = "testTrainExamplesValues.obj";
  
  AdversaryLearningConfiguration configuration;

  @Test
  void testNetUpdateLastIteration() throws IOException, InterruptedException {

    configuration =
        new AdversaryLearningConfiguration.Builder().
        alwaysUpdateNeuralNetwork(true).
        numberOfIterations(1).
        numberOfEpisodesBeforePotentialUpdate(1).
        bestModelFileName(TEST_MODEL_BIN).
        trainExamplesFileName(TEST_TRAIN_EXAMPLES).
        build();
    
    ComputationGraph computationGraph =
        new ComputationGraph(new ConvolutionResidualNet().createConvolutionalGraphConfiguration());
    computationGraph.init();
    
    AdversaryLearning learning = new AdversaryLearning(new TicTacToe(Game.MAX_PLAYER), computationGraph, configuration);
    NeuralNetUpdater neuralNetUpdater = new NeuralNetUpdater(
        new AdversaryLearningController(learning));
    
    assertEquals(0, computationGraph.getIterationCount());

    learning.performLearning();
    ExecutorService executorService = neuralNetUpdater.listenForNewTrainingExamples();
    executorService.shutdown();
    executorService.awaitTermination(2, TimeUnit.MINUTES);

    assertEquals(1, GraphLoader.loadComputationGraph(configuration).getIterationCount());
  }

  @Test
  void testMiniBatchBehavior() throws IOException, InterruptedException {
    
    configuration =
        new AdversaryLearningConfiguration.Builder().
        alwaysUpdateNeuralNetwork(true).
        numberOfIterations(1).
        numberOfEpisodesBeforePotentialUpdate(1).
        batchSize(16).
        bestModelFileName(TEST_MODEL_BIN).
        trainExamplesFileName(TEST_TRAIN_EXAMPLES).
        build();
 
    ComputationGraph computationGraph =
        new ComputationGraph(new ConvolutionResidualNet().createConvolutionalGraphConfiguration());
    computationGraph.init();
    
    AdversaryLearning learning = new AdversaryLearning(new TicTacToe(Game.MAX_PLAYER), computationGraph, configuration);
    NeuralNetUpdater neuralNetUpdater = new NeuralNetUpdater(
        new AdversaryLearningController(learning));
    
    assertEquals(0, computationGraph.getIterationCount());
    
    learning.performLearning();
    ExecutorService executorService = neuralNetUpdater.listenForNewTrainingExamples();
    executorService.shutdown();
    executorService.awaitTermination(2, TimeUnit.MINUTES);
    
    assertTrue(1 < GraphLoader.loadComputationGraph(configuration).getIterationCount());
  }

  @Test
  void testChallengeGames() throws IOException, InterruptedException {
    
    configuration =
        new AdversaryLearningConfiguration.Builder().
        alwaysUpdateNeuralNetwork(false).
        numberOfGamesToDecideUpdate(1).
        gamesWinRatioThresholdNewNetworkUpdate(-0.1).
        numberOfIterations(1).
        numberOfEpisodesBeforePotentialUpdate(1).
        bestModelFileName(TEST_MODEL_BIN).
        trainExamplesFileName(TEST_TRAIN_EXAMPLES).
        build();
 
    ComputationGraph computationGraph =
        new ComputationGraph(new ConvolutionResidualNet().createConvolutionalGraphConfiguration());
    computationGraph.init();
    
    AdversaryLearning learning = new AdversaryLearning(new TicTacToe(Game.MAX_PLAYER), computationGraph, configuration);
    NeuralNetUpdater neuralNetUpdater = new NeuralNetUpdater(
        new AdversaryLearningController(learning));
    
    assertEquals(0, computationGraph.getIterationCount());
    
    learning.performLearning();
    ExecutorService executorService = neuralNetUpdater.listenForNewTrainingExamples();
    executorService.shutdown();
    executorService.awaitTermination(2, TimeUnit.MINUTES);
    
    assertEquals(1, GraphLoader.loadComputationGraph(configuration).getIterationCount());

    Files.delete(Paths.get(configuration.getAbsolutePathFrom("tempmodel.bin")));
  }

  @AfterEach
  void deleteTempModel() throws IOException {

    Files.delete(Paths.get(configuration.getAbsolutePathFrom(TEST_MODEL_BIN)));
    Files.delete(Paths.get(configuration.getAbsolutePathFrom(TEST_TRAIN_EXAMPLES)));
    Files.delete(Paths.get(configuration.getAbsolutePathFrom(TEST_TRAIN_EXAMPLES_VALUES)));
  }
  
}
