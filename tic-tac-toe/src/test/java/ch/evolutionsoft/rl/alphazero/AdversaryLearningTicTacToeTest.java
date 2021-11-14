package ch.evolutionsoft.rl.alphazero;
/*
import static org.junit.jupiter.api.Assertions.*;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Collections;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.runner.RunWith;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.junit4.SpringRunner;

import ch.evolutionsoft.rl.AdversaryLearningConfiguration;
import ch.evolutionsoft.rl.Game;
import ch.evolutionsoft.rl.GraphLoader;
import ch.evolutionsoft.rl.alphazero.tictactoe.ConvolutionResidualNet;
import ch.evolutionsoft.rl.alphazero.tictactoe.TicTacToe;
import ch.evolutionsoft.rl.netupdate.NeuralNetUpdater;

@RunWith(SpringRunner.class)
@SpringBootTest(
    webEnvironment = SpringBootTest.WebEnvironment.DEFINED_PORT
)*/
class AdversaryLearningTicTacToeTest {
  /*
  public static final String TEST_MODEL_BIN = "testModel.bin";
  public static final String TEST_TRAIN_EXAMPLES = "testTrainExamples.obj";
  public static final String TEST_TRAIN_EXAMPLES_VALUES = "testTrainExamplesValues.obj";
  
  AdversaryLearningConfiguration baseConfiguration;

  @Autowired
  AdversaryLearning adversaryLearning;

  @BeforeEach
  void setupLearningConfiguration() {
    
    this.baseConfiguration =
        new AdversaryLearningConfiguration.Builder().
        learningRateSchedule(new MapSchedule(ScheduleType.ITERATION, Collections.singletonMap(0, 0.01))).
        alwaysUpdateNeuralNetwork(true).
        numberOfAllAvailableMoves(9).
        numberOfIterations(1).
        numberOfEpisodesBeforePotentialUpdate(1).
        bestModelFileName(TEST_MODEL_BIN).
        trainExamplesFileName(TEST_TRAIN_EXAMPLES).
        build();
  }
  
  @Test
  void testNetUpdateLastIteration() throws IOException, InterruptedException {
    
    ComputationGraph computationGraph =
        new ComputationGraph(new ConvolutionResidualNet().createConvolutionalGraphConfiguration());
    computationGraph.init();
    adversaryLearning.initialize(new TicTacToe(Game.MAX_PLAYER), computationGraph, baseConfiguration);
    adversaryLearning.performLearning();    
    assertEquals(0, computationGraph.getIterationCount());

    // Starting the NeuralNetUpdater will cause a neural net update
    NeuralNetUpdater.main(null);

    assertEquals(1, GraphLoader.loadComputationGraph(baseConfiguration).getIterationCount());
  }

  @Test
  void testMiniBatchBehavior() throws IOException, InterruptedException {
    
    baseConfiguration.setBatchSize(16);
    ComputationGraph computationGraph =
        new ComputationGraph(new ConvolutionResidualNet().createConvolutionalGraphConfiguration());
    computationGraph.init(); 
    adversaryLearning.initialize(new TicTacToe(Game.MAX_PLAYER), computationGraph, baseConfiguration);
    adversaryLearning.performLearning();    
    assertEquals(0, computationGraph.getIterationCount());

    // Starting the NeuralNetUpdater should cause several neural net updates with small batch size
    NeuralNetUpdater.main(null);
    
    assertTrue(1 < GraphLoader.loadComputationGraph(baseConfiguration).getIterationCount());
  }

  @Test
  void testChallengeGames() throws IOException, InterruptedException {
    
    baseConfiguration.setAlwaysUpdateNeuralNetwork(false);
    baseConfiguration.setNumberOfGamesToDecideUpdate(1);
    // Use negative ratio threshold always fullfilled
    baseConfiguration.setGamesWinRatioThresholdNewNetworkUpdate(-0.1);
    ComputationGraph computationGraph =
        new ComputationGraph(new ConvolutionResidualNet().createConvolutionalGraphConfiguration());
    computationGraph.init();
    adversaryLearning.initialize(new TicTacToe(Game.MAX_PLAYER), computationGraph, baseConfiguration);
    adversaryLearning.performLearning();    
    assertEquals(0, computationGraph.getIterationCount());

    // Starting the NeuralNetUpdater should cause net update after challenge games with any win ratio
    NeuralNetUpdater.main(null);
    
    assertEquals(1, GraphLoader.loadComputationGraph(baseConfiguration).getIterationCount());

    Files.delete(Paths.get(AdversaryLearningConfiguration.getAbsolutePathFrom("tempmodel.bin")));
  }

  @AfterEach
  void deleteTempModel() throws IOException {

    Files.delete(Paths.get(baseConfiguration.getAbsolutePathFrom(TEST_MODEL_BIN)));
    Files.delete(Paths.get(baseConfiguration.getAbsolutePathFrom(TEST_TRAIN_EXAMPLES)));
    Files.delete(Paths.get(baseConfiguration.getAbsolutePathFrom(TEST_TRAIN_EXAMPLES_VALUES)));
  }
  */
}
