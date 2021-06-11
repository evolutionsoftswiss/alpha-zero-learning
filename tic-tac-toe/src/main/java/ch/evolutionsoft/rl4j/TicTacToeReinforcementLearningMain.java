package ch.evolutionsoft.rl4j;

import java.io.IOException;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class TicTacToeReinforcementLearningMain {

  private static final Logger log = LoggerFactory.getLogger(TicTacToeReinforcementLearningMain.class);

  public static void main(String[] args) throws IOException {
    
    TicTacToeReinforcementLearningMain main = new TicTacToeReinforcementLearningMain();
   
    ComputationGraph neuralNet = main.createConvolutionalConfiguration();
    
    log.info(neuralNet.summary());
    
    AdversaryLearning adversaryLearning = new AdversaryLearning(neuralNet, 20);
    
    adversaryLearning.performLearning();
  }

  ComputationGraph createConvolutionalConfiguration() {

    ConvolutionResidualNet convolutionalLayerNet = new ConvolutionResidualNet(1e-3);

    ComputationGraphConfiguration convolutionalLayerNetConfiguration =
        convolutionalLayerNet.createConvolutionalGraphConfiguration();

    ComputationGraph net = new ComputationGraph(convolutionalLayerNetConfiguration);
    net.init();

    return net;
  }

}
