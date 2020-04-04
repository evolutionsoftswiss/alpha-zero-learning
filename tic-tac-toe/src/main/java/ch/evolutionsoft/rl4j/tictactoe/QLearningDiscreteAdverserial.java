package ch.evolutionsoft.rl4j.tictactoe;

import org.deeplearning4j.nn.api.NeuralNetwork;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.QLearningDiscrete;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.deeplearning4j.rl4j.policy.EpsGreedy;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.util.DataManager;

import ch.evolutionsoft.rl4j.tictactoe.ReinforcementLearningMain.TicTacToeState;

public class QLearningDiscreteAdverserial extends QLearningDiscrete<TicTacToeState> {
  
  TicTacToeEpsilonGreedyPolicy policy;
  NeuralNetwork perfectPlayingModel; 

  public QLearningDiscreteAdverserial(TicTacToeGame mdp, IDQN neuralNet,
      QLConfiguration qLConfiguration, MultiLayerNetwork perfectModel) {

    this(mdp, neuralNet, qLConfiguration, qLConfiguration.getEpsilonNbStep());
    
    this.perfectPlayingModel = perfectModel;
    policy = new TicTacToeEpsilonGreedyPolicy(mdp, qLConfiguration, perfectPlayingModel, this); 
  }

  public QLearningDiscreteAdverserial(
      MDP<TicTacToeState, Integer, DiscreteSpace> mdp,
      IDQN dqn,
      QLConfiguration conf,
      int epsilonNbStep) {

    super(mdp, dqn, conf, epsilonNbStep);
  }

  @Override
  public EpsGreedy<TicTacToeState, Integer, DiscreteSpace> getEgPolicy() {

    return policy;
  }
  
}
