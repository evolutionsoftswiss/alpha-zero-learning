package ch.evolutionsoft.rl.alphazero.connectfour;

import java.util.HashMap;
import java.util.Map;

import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;

import ch.evolutionsoft.rl.AdversaryLearningConfiguration;
import ch.evolutionsoft.rl.Game;
import ch.evolutionsoft.rl.alphazero.connectfour.model.PlaygroundConstants;

public class ConnectFourConfiguration {

  public static AdversaryLearningConfiguration getTrainingConfiguration(Game connectFourGame) {

    Map<Integer, Double> learningRatesByIterations = new HashMap<>();
    learningRatesByIterations.put(0, 1e-4);
    learningRatesByIterations.put(40000, 5e-5);
    learningRatesByIterations.put(100000, 1e-5);
    MapSchedule learningRateMapSchedule = new MapSchedule(ScheduleType.ITERATION, learningRatesByIterations);
 
    return 
        new AdversaryLearningConfiguration.Builder().
        learningRateSchedule(learningRateMapSchedule).
        numberOfAllAvailableMoves(connectFourGame.getNumberOfAllAvailableMoves()).
        batchSize(4096).
        checkPointIterationsFrequency(50).
        dirichletAlpha(0.7).
        dirichletWeight(0.35).
        fromNumberOfIterationsReducedTemperature(-1).
        fromNumberOfMovesReducedTemperature(-1).
        continueTraining(false).
        maxTrainExamplesHistory(81920).
        maxTrainExamplesHistoryFromIteration(300).
        numberOfIterations(4000).
        numberOfEpisodesBeforePotentialUpdate(20).
        numberOfEpisodeThreads(20).
        numberOfMonteCarloSimulations(200).
        uctConstantFactor(2.5).
        
        build();
  }

  public static AdversaryLearningConfiguration getDefaultPlayConfiguration() {
 
    return 
        new AdversaryLearningConfiguration.Builder().
        numberOfAllAvailableMoves(PlaygroundConstants.COLUMN_COUNT).
        numberOfMonteCarloSimulations(200).
        uctConstantFactor(2.5).
        
        build();
  }

  public static AdversaryLearningConfiguration getExplorativePlayConfiguration() {
 
    return 
        new AdversaryLearningConfiguration.Builder().
        numberOfAllAvailableMoves(PlaygroundConstants.COLUMN_COUNT).
        numberOfMonteCarloSimulations(200).
        uctConstantFactor(3.5).
        
        build();
  }
  
  private ConnectFourConfiguration() {
    // Hide constructor
  }
  
}
