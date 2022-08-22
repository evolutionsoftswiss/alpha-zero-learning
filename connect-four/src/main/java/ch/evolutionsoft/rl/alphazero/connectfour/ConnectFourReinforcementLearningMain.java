package ch.evolutionsoft.rl.alphazero.connectfour;

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
import org.springframework.context.ConfigurableApplicationContext;
import org.springframework.stereotype.Component;

import ch.evolutionsoft.rl.AdversaryLearningConfiguration;
import ch.evolutionsoft.rl.Game;
import ch.evolutionsoft.rl.alphazero.AdversaryLearning;
import ch.evolutionsoft.rl.alphazero.AdversaryLearningController;

@Component
public class ConnectFourReinforcementLearningMain {

  private static final Logger log = LoggerFactory.getLogger(ConnectFourReinforcementLearningMain.class);

  @Autowired
  AdversaryLearning adversaryLearning;
  
  public static void main(String[] args) throws IOException {

    ConfigurableApplicationContext applicationContext = SpringApplication.run(AdversaryLearningController.class);
    
    ConnectFourReinforcementLearningMain mainClass = applicationContext.getBean(ConnectFourReinforcementLearningMain.class);
    
    mainClass.adversaryLearning.performLearning();
  }

  @PostConstruct
  public void init() {
 
    ConnectFour connectFourGame = new ConnectFour(Game.MAX_PLAYER);

    Map<Integer, Double> learningRatesByIterations = new HashMap<>();
    learningRatesByIterations.put(0, 5e-4);
    learningRatesByIterations.put(40000, 2e-4);
    learningRatesByIterations.put(100000, 8e-5);
    MapSchedule learningRateMapSchedule = new MapSchedule(ScheduleType.ITERATION, learningRatesByIterations);
    AdversaryLearningConfiguration adversaryLearningConfiguration =
        new AdversaryLearningConfiguration.Builder().
        learningRateSchedule(learningRateMapSchedule).
        alwaysUpdateNeuralNetwork(true).
        numberOfAllAvailableMoves(connectFourGame.getNumberOfAllAvailableMoves()).
        batchSize(4096).
        checkPointIterationsFrequency(50).
        dirichletAlpha(2.0).
        dirichletWeight(0.35).
        fromNumberOfIterationsReducedTemperature(-1).
        fromNumberOfMovesReducedTemperature(-1).
        iterationStart(1).
        maxTrainExamplesHistory(81920).
        maxTrainExamplesHistoryFromIteration(300).
        numberOfIterations(4000).
        numberOfEpisodesBeforePotentialUpdate(20).
        numberOfEpisodeThreads(20).
        numberOfMonteCarloSimulations(200).
        uctConstantFactor(2.5).
        
        build();
   
    ComputationGraph neuralNet = createConvolutionalConfiguration(adversaryLearningConfiguration);

    if (log.isInfoEnabled()) {
      log.info(neuralNet.summary());
    }
    
    adversaryLearning.initialize(
            connectFourGame,
            neuralNet,
            adversaryLearningConfiguration);
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
