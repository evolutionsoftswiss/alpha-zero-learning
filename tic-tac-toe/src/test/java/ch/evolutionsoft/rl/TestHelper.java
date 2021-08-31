package ch.evolutionsoft.rl;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;

import ch.evolutionsoft.rl.tictactoe.ConvolutionResidualNet;
import ch.evolutionsoft.rl.tictactoe.TicTacToe;

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
  public static Game createMiddlePositionBoardWithThreat() {

    Game game = new TicTacToe(Game.MAX_PLAYER);

    game.makeMove(0, Game.MAX_PLAYER);
    game.makeMove(4, Game.MIN_PLAYER);
    game.makeMove(3, Game.MAX_PLAYER);
    game.makeMove(6, Game.MIN_PLAYER);
    game.makeMove(2, Game.MAX_PLAYER);

    return game;
  }
  
  private TestHelper() {
    // Hide constructor
  }

}
