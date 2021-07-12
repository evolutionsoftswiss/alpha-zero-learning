package ch.evolutionsoft.rl.tictactoe;

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
import ch.evolutionsoft.rl.Game;

public class TicTacToeReinforcementLearningMain {

  private static final Logger log = LoggerFactory.getLogger(TicTacToeReinforcementLearningMain.class);

  public static void main(String[] args) throws IOException {
    
    TicTacToeReinforcementLearningMain main = new TicTacToeReinforcementLearningMain();
    
    Map<Integer, Double> learningRatesByIterations = new HashMap<>();
    learningRatesByIterations.put(0, 8e-4);
    learningRatesByIterations.put(500, 1e-4);
    MapSchedule learningRateMapSchedule = new MapSchedule(ScheduleType.ITERATION, learningRatesByIterations);
    AdversaryLearningConfiguration adversaryLearningConfiguration =
        new AdversaryLearningConfiguration.Builder().learningRateSchedule(learningRateMapSchedule).
        build();
   
    ComputationGraph neuralNet = main.createConvolutionalConfiguration(adversaryLearningConfiguration);
    
    log.info(neuralNet.summary());
    
    AdversaryLearning adversaryLearning =
        new AdversaryLearning(
            new TicTacToe(Game.MAX_PLAYER),
            neuralNet,
            adversaryLearningConfiguration);
    
    adversaryLearning.performLearning();
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
