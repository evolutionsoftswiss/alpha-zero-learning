package ch.evolutionsoft.rl.alphazero.tictactoe;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import javax.annotation.PostConstruct;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.SpringApplication;
import org.springframework.stereotype.Component;

import ch.evolutionsoft.rl.AdversaryLearningConfiguration;
import ch.evolutionsoft.rl.Game;
import ch.evolutionsoft.rl.alphazero.AdversaryLearning;
import ch.evolutionsoft.rl.alphazero.AdversaryLearningController;

@Component
public class TicTacToeReinforcementLearningMain {

  private static final Logger log = LoggerFactory.getLogger(TicTacToeReinforcementLearningMain.class);

  @Autowired
  AdversaryLearning adversaryLearning;
  
  public static void main(String[] args) {

    SpringApplication.run(AdversaryLearningController.class);
  }

  @PostConstruct
  public void init() throws IOException {

    TicTacToe tictactoeGame = new TicTacToe(Game.MAX_PLAYER);
    
    Map<Integer, Double> learningRatesByIterations = new HashMap<>();
    learningRatesByIterations.put(0, 2e-3);
    learningRatesByIterations.put(200, 1e-3);
    MapSchedule learningRateMapSchedule = new MapSchedule(ScheduleType.ITERATION, learningRatesByIterations);
    AdversaryLearningConfiguration adversaryLearningConfiguration =
        new AdversaryLearningConfiguration.Builder().
        learningRateSchedule(learningRateMapSchedule).
        numberOfAllAvailableMoves(tictactoeGame.getNumberOfAllAvailableMoves()).
        build();
   
    ComputationGraph neuralNet = createConvolutionalConfiguration(adversaryLearningConfiguration);

    if (log.isInfoEnabled()) {
      log.info(neuralNet.summary());
    }
    
    adversaryLearning.initialize(
            tictactoeGame,
            neuralNet,
            adversaryLearningConfiguration);

    adversaryLearning.performLearning();
  }

  ComputationGraph createConvolutionalConfiguration(AdversaryLearningConfiguration adversaryLearningConfiguration) {

    ConvolutionResidualNet convolutionalLayerNet =
        new ConvolutionResidualNet(adversaryLearningConfiguration.getLearningRateSchedule());
    
    ComputationGraphConfiguration convolutionalLayerNetConfiguration =
        convolutionalLayerNet.createConvolutionalGraphConfiguration();

    ComputationGraph net = new ComputationGraph(convolutionalLayerNetConfiguration);
    net.init();

    return net;
  }
}
