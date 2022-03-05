package ch.evolutionsoft.rl;

import static org.junit.jupiter.api.Assertions.*;

import java.io.File;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;
import org.opentest4j.MultipleFailuresError;

class AdversaryLearningConfigurationTest {

  @Test
  void checkConfigurationBuilderValues() {

    Map<Integer, Double> learningRatesByIteration = new HashMap<>();
    learningRatesByIteration.put(0, 1e-4);
    MapSchedule learningRateSchedule = new MapSchedule(ScheduleType.ITERATION, learningRatesByIteration);
    
    AdversaryLearningConfiguration adversaryLearningConfiguration =
        new AdversaryLearningConfiguration.Builder().
        alwaysUpdateNeuralNetwork(false).
        numberOfAllAvailableMoves(7).
        batchSize(1024).
        bestModelFileName("alphaModel.bin").
        checkPointIterationsFrequency(5).
        dirichletAlpha(0.8).
        dirichletWeight(0.4).
        fromNumberOfIterationsTemperatureZero(1000).
        fromNumberOfMovesTemperatureZero(10).
        gamesWinRatioThresholdNewNetworkUpdate(0.5).
        iterationStart(10000).
        learningRateSchedule(learningRateSchedule).
        maxTrainExamplesHistory(100000).
        numberOfGamesToDecideUpdate(50).
        numberOfIterations(1000).
        numberOfEpisodesBeforePotentialUpdate(20).
        numberOfMonteCarloSimulations(100).
        trainExamplesFileName("trainingExamplesHistory.obj").
        uctConstantFactor(1.4).
        build();
    
    assertAllConfigurationExpectedValues(learningRateSchedule, adversaryLearningConfiguration);
  }

  @Test
  void checkConfigurationSetterValues() {

    Map<Integer, Double> learningRatesByIteration = new HashMap<>();
    learningRatesByIteration.put(0, 1e-4);
    MapSchedule learningRateSchedule = new MapSchedule(ScheduleType.ITERATION, learningRatesByIteration);
    
    AdversaryLearningConfiguration adversaryLearningConfiguration = new AdversaryLearningConfiguration();
    adversaryLearningConfiguration.setAlwaysUpdateNeuralNetwork(false);
    adversaryLearningConfiguration.setNumberOfAllAvailableMoves(7);
    adversaryLearningConfiguration.setBatchSize(1024);
    adversaryLearningConfiguration.setBestModelFileName("alphaModel.bin");
    adversaryLearningConfiguration.setCheckPointIterationsFrequency(5);
    adversaryLearningConfiguration.setDirichletAlpha(0.8);
    adversaryLearningConfiguration.setDirichletWeight(0.4);
    adversaryLearningConfiguration.setFromNumberOfIterationsTemperatureZero(1000);
    adversaryLearningConfiguration.setFromNumberOfMovesTemperatureZero(10);
    adversaryLearningConfiguration.setGamesWinRatioThresholdNewNetworkUpdate(0.5);
    adversaryLearningConfiguration.setIterationStart(10000);
    adversaryLearningConfiguration.setLearningRateSchedule(learningRateSchedule);
    adversaryLearningConfiguration.setMaxTrainExamplesHistory(100000);
    adversaryLearningConfiguration.setNumberOfGamesToDecideUpdate(50);
    adversaryLearningConfiguration.setNumberOfIterations(1000);
    adversaryLearningConfiguration.setNumberOfEpisodesBeforePotentialUpdate(20);
    adversaryLearningConfiguration.setNumberOfMonteCarloSimulations(100);
    adversaryLearningConfiguration.setTrainExamplesFileName("trainingExamplesHistory.obj");
    adversaryLearningConfiguration.setUctConstantFactor(1.4);
    
    assertAllConfigurationExpectedValues(learningRateSchedule, adversaryLearningConfiguration);
  }

  void assertAllConfigurationExpectedValues(MapSchedule learningRateSchedule,
      AdversaryLearningConfiguration adversaryLearningConfiguration) throws MultipleFailuresError {

    assertAll(
        () -> assertEquals(String.valueOf(Paths.get("").toAbsolutePath()) + File.separator + "alphaModel.bin", 
            AdversaryLearningConfiguration.getAbsolutePathFrom(adversaryLearningConfiguration.getBestModelFileName())),
        () -> assertEquals(false, adversaryLearningConfiguration.isAlwaysUpdateNeuralNetwork()),
        () -> assertEquals(7, adversaryLearningConfiguration.getNumberOfAllAvailableMoves()),
        () -> assertEquals(1024, adversaryLearningConfiguration.getBatchSize()),
        () -> assertTrue(adversaryLearningConfiguration.getBestModelFileName().endsWith("alphaModel.bin")),
        () -> assertTrue(adversaryLearningConfiguration.getTrainExamplesFileName().endsWith("trainingExamplesHistory.obj")),
        () -> assertEquals(5, adversaryLearningConfiguration.getCheckPointIterationsFrequency()),
        () -> assertEquals(1.0, adversaryLearningConfiguration.getCurrentTemperature(0, 9)),
        () -> assertEquals(0.0, adversaryLearningConfiguration.getCurrentTemperature(0, 10)),
        () -> assertEquals(1.0, adversaryLearningConfiguration.getCurrentTemperature(999, 1)),
        () -> assertEquals(0.0, adversaryLearningConfiguration.getCurrentTemperature(1000, 1)),
        () -> assertEquals(0.0, adversaryLearningConfiguration.getCurrentTemperature(1000, 10)),
        () -> assertEquals(0.8, adversaryLearningConfiguration.getDirichletAlpha()),
        () -> assertEquals(0.4, adversaryLearningConfiguration.getDirichletWeight()),
        () -> assertEquals(1000, adversaryLearningConfiguration.getFromNumberOfIterationsTemperatureZero()),
        () -> assertEquals(10, adversaryLearningConfiguration.getFromNumberOfMovesTemperatureZero()),
        () -> assertEquals(0.5, adversaryLearningConfiguration.getGamesWinRatioThresholdNewNetworkUpdate()),
        () -> assertEquals(10000, adversaryLearningConfiguration.getIterationStart()),
        () -> assertEquals(learningRateSchedule, adversaryLearningConfiguration.getLearningRateSchedule()),
        () -> assertEquals(100000, adversaryLearningConfiguration.getMaxTrainExamplesHistory()),
        () -> assertEquals(50, adversaryLearningConfiguration.getNumberOfGamesToDecideUpdate()),
        () -> assertEquals(1000, adversaryLearningConfiguration.getNumberOfIterations()),
        () -> assertEquals(20, adversaryLearningConfiguration.getNumberOfEpisodesBeforePotentialUpdate()),
        () -> assertEquals(100, adversaryLearningConfiguration.getNumberOfMonteCarloSimulations()),
        () -> assertEquals(1.4, adversaryLearningConfiguration.getuctConstantFactor())
    );
  }

}
