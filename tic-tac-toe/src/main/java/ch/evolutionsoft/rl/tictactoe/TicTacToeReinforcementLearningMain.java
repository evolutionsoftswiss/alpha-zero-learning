package ch.evolutionsoft.rl.tictactoe;

import java.io.IOException;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.schedule.ISchedule;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ch.evolutionsoft.rl.AdversaryLearning;
import ch.evolutionsoft.rl.AdversaryLearningConfiguration;
import ch.evolutionsoft.rl.Game;

public class TicTacToeReinforcementLearningMain {

  private static final Logger log = LoggerFactory.getLogger(TicTacToeReinforcementLearningMain.class);

  public static void main(String[] args) throws IOException {
    
    TicTacToeReinforcementLearningMain main = new TicTacToeReinforcementLearningMain();
    
    AdversaryLearningConfiguration adversaryLearningConfiguration =
        new AdversaryLearningConfiguration.Builder().
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

    ComputationGraphConfiguration convolutionalLayerNetConfiguration =
        convolutionalLayerNet.createConvolutionalGraphConfiguration();

    ComputationGraph net = new ComputationGraph(convolutionalLayerNetConfiguration);
    net.init();

    return net;
  }
  
  static class TicTacToeLearningShedule implements ISchedule {
    
    @Override
    public double valueAt(int iteration, int epoch) {

      if (iteration <= 600) {
        return 8e-4;
      }
      
      return 1e-4;
    }

    @Override
    public ISchedule clone() {
      return new TicTacToeLearningShedule();
    }
    
  }
}
