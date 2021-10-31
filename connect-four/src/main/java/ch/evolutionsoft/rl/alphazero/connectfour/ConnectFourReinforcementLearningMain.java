package ch.evolutionsoft.rl.alphazero.connectfour;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ch.evolutionsoft.rl.AdversaryLearning;
import ch.evolutionsoft.rl.AdversaryLearningConfiguration;
import ch.evolutionsoft.rl.AdversaryLearningController;
import ch.evolutionsoft.rl.Game;
import ch.evolutionsoft.rl.NeuralNetUpdater;

public class ConnectFourReinforcementLearningMain {

  private static final Logger log = LoggerFactory.getLogger(ConnectFourReinforcementLearningMain.class);

  public static void main(String[] args) throws IOException {
    
    ConnectFourReinforcementLearningMain main = new ConnectFourReinforcementLearningMain();
    
    Map<Integer, Double> learningRatesByIterations = new HashMap<>();
    learningRatesByIterations.put(0, 2e-3);
    learningRatesByIterations.put(2000, 1e-3);
    learningRatesByIterations.put(4000, 5e-4);
    MapSchedule learningRateMapSchedule = new MapSchedule(ScheduleType.ITERATION, learningRatesByIterations);
    AdversaryLearningConfiguration adversaryLearningConfiguration =
        new AdversaryLearningConfiguration.Builder().
        learningRateSchedule(learningRateMapSchedule).
        alwaysUpdateNeuralNetwork(true).
        batchSize(32768).
        checkPointIterationsFrequency(50).
        dirichletAlpha(1.1).
        dirichletWeight(0.4).
        fromNumberOfIterationsTemperatureZero(-1).
        fromNumberOfMovesTemperatureZero(10).
        iterationStart(1001).
        maxTrainExamplesHistory(80000).
        numberOfIterations(100).
        numberOfEpisodesBeforePotentialUpdate(10).
        numberOfEpisodeThreads(10).
        numberOfMonteCarloSimulations(200).
        uctConstantFactor(1.5).
        
        build();
   
    ComputationGraph neuralNet = main.createConvolutionalConfiguration(adversaryLearningConfiguration);

    if (log.isInfoEnabled()) {
      log.info(neuralNet.summary());
    }
    
    AdversaryLearning adversaryLearning =
        new AdversaryLearning(
            new ConnectFour(Game.MAX_PLAYER),
            neuralNet,
            adversaryLearningConfiguration);
    
    adversaryLearning.performLearning();
    
    NeuralNetUpdater neuralNetUpdater = new NeuralNetUpdater(
        new AdversaryLearningController(adversaryLearning));
    
    neuralNetUpdater.listenForNewTrainingExamples();
  }

  ComputationGraph createConvolutionalConfiguration(AdversaryLearningConfiguration adversaryLearningConfiguration) {

    ConvolutionResidualNet convolutionalLayerNet =
        new ConvolutionResidualNet(adversaryLearningConfiguration.getLearningRate());

    if (null != adversaryLearningConfiguration.getLearningRateSchedule()) {

      convolutionalLayerNet =
          new ConvolutionResidualNet(adversaryLearningConfiguration.getLearningRateSchedule());
    }
    
    ComputationGraphConfiguration convolutionalLayerNetConfiguration =
        convolutionalLayerNet.createConvolutionalGraphConfiguration();

    ComputationGraph net = new ComputationGraph(convolutionalLayerNetConfiguration);
    net.init();

    return net;
  }

}
