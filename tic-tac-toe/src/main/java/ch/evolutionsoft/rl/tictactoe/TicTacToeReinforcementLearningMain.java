package ch.evolutionsoft.rl.tictactoe;

import java.io.IOException;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ch.evolutionsoft.rl.AdversaryLearning;
import ch.evolutionsoft.rl.AdversaryLearningConfiguration;
import ch.evolutionsoft.rl.Game;

public class TicTacToeReinforcementLearningMain {

  private static final Logger log = LoggerFactory.getLogger(TicTacToeReinforcementLearningMain.class);

  public static void main(String[] args) throws IOException {
    
    TicTacToeReinforcementLearningMain main = new TicTacToeReinforcementLearningMain();
   
    ComputationGraph neuralNet = main.createConvolutionalConfiguration();
    
    log.info(neuralNet.summary());
    
    AdversaryLearning adversaryLearning =
        new AdversaryLearning(
            new TicTacToe(Game.MAX_PLAYER),
            neuralNet,
            new AdversaryLearningConfiguration.Builder().
            build());
    
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
