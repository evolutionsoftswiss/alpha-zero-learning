package ch.evolutionsoft.rl.alphazero.connectfour;

import java.io.IOException;

import javax.annotation.PostConstruct;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
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
  public void init() throws IOException {
 
    Game connectFourGame = new BinaryConnectFour(Game.MAX_PLAYER);

    AdversaryLearningConfiguration adversaryLearningConfiguration =
        ConnectFourConfiguration.getTrainingConfiguration(connectFourGame);
   
    ComputationGraph neuralNet = createConvolutionalConfiguration(adversaryLearningConfiguration);

    if (log.isInfoEnabled()) {
      log.info(neuralNet.summary());
    }
    
    adversaryLearning.initialize(
            connectFourGame,
            neuralNet,
            adversaryLearningConfiguration);
  }
  
  ComputationGraph createConvolutionalConfiguration(AdversaryLearningConfiguration adversaryLearningConfiguration) throws IOException {

    ConvolutionResidualNet convolutionalLayerNet =
        new ConvolutionResidualNet(adversaryLearningConfiguration.getLearningRateSchedule());

    ComputationGraphConfiguration convolutionalLayerNetConfiguration =
        convolutionalLayerNet.createConvolutionalGraphConfiguration();
        
    ComputationGraph net = new ComputationGraph(convolutionalLayerNetConfiguration);
    net.init();
    
    return net;
  }
}
