package ch.evolutionsoft.rl4j;

import java.io.IOException;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;

public class TicTacToeReinforcementLearningMain {

  public static void main(String[] args) throws IOException {
    
    TicTacToeReinforcementLearningMain main = new TicTacToeReinforcementLearningMain();
   
    ComputationGraph neuralNet = main.createConvolutionalConfiguration();
    
    AdversaryLearning adversaryLearning = new AdversaryLearning(neuralNet, 50);
    
    adversaryLearning.performLearning();
  }

  ComputationGraph createConvolutionalConfiguration() {

    ConvolutionResidualNet convolutionalLayerNet = new ConvolutionResidualNet(5e-3);

    ComputationGraphConfiguration convolutionalLayerNetConfiguration =
        convolutionalLayerNet.createConvolutionalGraphConfiguration();

    ComputationGraph net = new ComputationGraph(convolutionalLayerNetConfiguration);
    net.init();

    return net;
  }

}
