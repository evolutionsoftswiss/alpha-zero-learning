package ch.evolutionsoft.rl4j.tictactoe;

import org.deeplearning4j.nn.api.NeuralNetwork;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.learning.StepCountable;
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning.QLConfiguration;
import org.deeplearning4j.rl4j.network.NeuralNet;
import org.deeplearning4j.rl4j.policy.EpsGreedy;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.rng.CpuNativeRandom;
import org.nd4j.linalg.factory.Nd4j;

import ch.evolutionsoft.net.game.NeuralNetConstants;
import ch.evolutionsoft.rl4j.tictactoe.ReinforcementLearningMain.TicTacToeAction;
import ch.evolutionsoft.rl4j.tictactoe.ReinforcementLearningMain.TicTacToeState;

public class TicTacToeEpsilonGreedyPolicy extends EpsGreedy<TicTacToeState, Integer, DiscreteSpace> {

  final protected TicTacToeGame mdp;
  final protected int updateStart;
  final protected int epsilonNbStep;
  final protected float minEpsilon;
  final protected StepCountable stepCountable;

  final private NeuralNetwork perfectPlayer;

  public float getEpsilon() {

    return Math.min(1f, Math.max(minEpsilon, 1f - (stepCountable.getStepCounter() - updateStart) * 1f / epsilonNbStep));
  }

  public TicTacToeEpsilonGreedyPolicy(TicTacToeGame mdp, QLConfiguration qLConfiguration,
      NeuralNetwork perfectPlayingModel, StepCountable stepCountable) {

    super(null, null, qLConfiguration.getUpdateStart(), qLConfiguration.getEpsilonNbStep(),
        new CpuNativeRandom(NeuralNetConstants.DEFAULT_SEED), qLConfiguration.getMinEpsilon(), stepCountable);

    this.mdp = mdp;
    this.updateStart = qLConfiguration.getUpdateStart();
    this.epsilonNbStep = qLConfiguration.getEpsilonNbStep();
    this.minEpsilon = qLConfiguration.getMinEpsilon();
    this.stepCountable = stepCountable;
    this.perfectPlayer = perfectPlayingModel;
  }

  @Override
  public Integer nextAction(INDArray input) {

    float ep = getEpsilon();

    if (mdp.getCurrentPlayerChannel(input) == TicTacToeGame.TRAINING_PLAYER) {

      if (NeuralNetConstants.randomGenerator.nextFloat() > ep) {

        return nextLegalAction(input, mdp.getActionSpace(input));
      }

      return mdp.getActionSpace(input).randomAction();
    }

    if (mdp.allFieldsEmpty(input)) {

      return mdp.getActionSpace(input).randomAction();
    }

    INDArray perfectOutput;
    if (perfectPlayer instanceof ComputationGraph) {

      perfectOutput = ((ComputationGraph) perfectPlayer).outputSingle(input);

    } else {

      perfectOutput = ((MultiLayerNetwork) perfectPlayer).output(input);
    }

    return Learning.getMaxAction(perfectOutput);
  }

  public Integer nextLegalAction(INDArray input, TicTacToeAction legalAction) {

    INDArray output = mdp.getFetchable().getNeuralNet().output(input);

    INDArray legalOutput = output.dup();
    int columnNumber = legalOutput.columns();
    for (int column = 0; column < columnNumber; column++) {

      if (!legalAction.availableMoves.contains(column)) {

        legalOutput.putScalar(column, -1);

      }
    }

    /*
     * if (TicTacToeGameHelper.getCurrentPlayer(input) == MAX_PLAYER) {
     * 
     * if (Nd4j.max(legalOutput).getDouble(0) == ZERO) {
     * 
     * return
     * legalAction.availableMoves.get(randomGenerator.nextInt(legalAction.
     * availableMoves.size())); }
     */

    return Nd4j.argMax(legalOutput, Integer.MAX_VALUE).getInt(0);
    // }

    /*
     * INDArray minimizedMax = legalOutput.muli(Integer.valueOf(-1));
     * 
     * if (Nd4j.max(minimizedMax).getDouble(0) == ZERO) {
     * 
     * return
     * legalAction.availableMoves.get(randomGenerator.nextInt(legalAction.
     * availableMoves.size())); }
     * 
     * return Nd4j.argMax(minimizedMax, Integer.MAX_VALUE).getInt(0);
     */
  }

  public NeuralNet getNeuralNet() {
    return mdp.getFetchable().getNeuralNet();
  }

}
