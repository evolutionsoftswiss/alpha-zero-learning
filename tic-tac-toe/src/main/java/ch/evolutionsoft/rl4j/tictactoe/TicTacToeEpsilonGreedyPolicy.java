package ch.evolutionsoft.rl4j.tictactoe;

import org.deeplearning4j.nn.graph.ComputationGraph;
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

  protected final TicTacToeGame mdp;
  protected final int updateStart;
  protected final int epsilonNbStep;
  protected final float minEpsilon;
  protected final StepCountable stepCountable;

  private final ComputationGraph perfectPlayer;

  public TicTacToeEpsilonGreedyPolicy(TicTacToeGame mdp, QLConfiguration qLConfiguration,
      ComputationGraph perfectPlayingModel, StepCountable stepCountable) {

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
    
    // Here input is batched shaped
    INDArray reducedInput = input.dup().slice(0);

    float ep = getEpsilon();

    if (mdp.getCurrentPlayer(reducedInput) == mdp.getTrainingPlayer()) {

      if (NeuralNetConstants.randomGenerator.nextFloat() > ep) {

        return nextLegalAction(input, mdp.getActionSpace(reducedInput));
      }

      return mdp.getActionSpace(reducedInput).randomAction();
    }

    if (mdp.allFieldsEmpty(reducedInput)) {

      return mdp.getActionSpace(reducedInput).randomAction();
    }

    INDArray perfectOutput = perfectPlayer.outputSingle(input);
    
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

    return Nd4j.argMax(legalOutput, Integer.MAX_VALUE).getInt(0);
  }

  @Override
  public NeuralNet<ConvolutionalNeuralNetDQN> getNeuralNet() {
    return mdp.getFetchable().getNeuralNet();
  }

  @Override
  public float getEpsilon() {

    return Math.min(1f, Math.max(minEpsilon, 1f - (stepCountable.getStepCounter() - updateStart) * 1f / epsilonNbStep));
  }

}
