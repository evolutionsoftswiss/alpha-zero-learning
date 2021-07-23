package ch.evolutionsoft.rl;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;

import ch.evolutionsoft.net.game.NeuralNetConstants;
import ch.evolutionsoft.net.game.tictactoe.TicTacToeConstants;
import ch.evolutionsoft.rl.tictactoe.ConvolutionResidualNet;

public class TestHelper {

  public static ComputationGraph createConvolutionalConfiguration() {

    ConvolutionResidualNet convolutionalLayerNet =
        new ConvolutionResidualNet();
    
    ComputationGraphConfiguration convolutionalLayerNetConfiguration =
        convolutionalLayerNet.createConvolutionalGraphConfiguration();

    ComputationGraph net = new ComputationGraph(convolutionalLayerNetConfiguration);
    net.init();

    return net;
  }

  /**
   * 
   * X| |X
   * X|O| 
   * O| | 
   * 
   * @return
   */
  public static INDArray createMiddlePositionBoardWithThreat() {
    
    INDArray moveFiveBoard = TicTacToeConstants.EMPTY_CONVOLUTIONAL_PLAYGROUND;
    moveFiveBoard.putScalar(TicTacToeConstants.MAX_PLAYER_CHANNEL, 0, 0, NeuralNetConstants.ONE);
    moveFiveBoard.putScalar(TicTacToeConstants.MAX_PLAYER_CHANNEL, 1, 0, NeuralNetConstants.ONE);
    moveFiveBoard.putScalar(TicTacToeConstants.MAX_PLAYER_CHANNEL, 0, 2, NeuralNetConstants.ONE);
    moveFiveBoard.putScalar(TicTacToeConstants.MIN_PLAYER_CHANNEL, 1, 1, NeuralNetConstants.ONE);
    moveFiveBoard.putScalar(TicTacToeConstants.MIN_PLAYER_CHANNEL, 2, 0, NeuralNetConstants.ONE);
    moveFiveBoard.putRow(TicTacToeConstants.CURRENT_PLAYER_CHANNEL, moveFiveBoard.slice(TicTacToeConstants.CURRENT_PLAYER_CHANNEL).mul(-1));
    
    return moveFiveBoard;
  }
  
  private TestHelper() {
    // Hide constructor
  }

}
