package ch.evolutionsoft.rl4j.tictactoe;

import static ch.evolutionsoft.net.game.tictactoe.TicTacToeConstants.*;
import static ch.evolutionsoft.net.game.NeuralNetConstants.*;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration.Builder;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.rl4j.learning.NeuralNetFetchable;
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning.QLConfiguration;
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.QLearningDiscrete;
import org.deeplearning4j.rl4j.network.dqn.DQN;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.util.DataManager;
import org.deeplearning4j.rl4j.util.IDataManager.StatEntry;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ch.evolutionsoft.net.game.NeuralDataHelper;
import ch.evolutionsoft.net.game.tictactoe.TicTacToeConstants;
import ch.evolutionsoft.net.game.tictactoe.TicTacToeNeuralDataConverter;

public class ReinforcementLearningMain {

  private static final int NUMBER_OF_NODES = 42;

  Pair<INDArray, INDArray> p1 = TicTacToeNeuralDataConverter.stackFeedForwardPlaygroundLabels(
      TicTacToeNeuralDataConverter.convertMiniMaxLabels(
          NeuralDataHelper.readAll("/inputsMin.txt", "/labelsMin.txt")));

  QLearningDiscrete<TicTacToeState> dql;

  List<StatEntry> statisticEntries = new LinkedList<>();

  static class TicTacToeState implements Encodable {

    private INDArray playground = EMPTY_PLAYGROUND;

    int depth = 0;

    public TicTacToeState(INDArray playground, int depth) {

      this.playground = playground;
      this.depth = depth;
    }

    public INDArray getPlayground() {
      return playground.dup();
    }

    public TicTacToeState makeMove(int flatIndex) {

      INDArray newPlayground = this.getPlayground();
      newPlayground.putScalar(0, flatIndex, getCurrentPlayerChannel());

      return new TicTacToeState(newPlayground, ++this.depth);
    }

    public int getCurrentPlayerChannel() {

      if (depth % 2 == 0) {

        return MAX_PLAYER_CHANNEL;
      }

      return MIN_PLAYER_CHANNEL;
    }

    @Override
    public double[] toArray() {

      return getPlayground().reshape(1, COLUMN_NUMBER).toDoubleVector();
    }
  }

  static class TicTacToeAction extends DiscreteSpace {

    Set<Integer> availableMoves;

    public TicTacToeAction(Set<Integer> availableMoves) {

      super(availableMoves.size());
      this.availableMoves = new HashSet<>(availableMoves);
    }

    public Integer randomAction() {

      List<Integer> emptyIndicesList = new ArrayList<>(availableMoves);
      return emptyIndicesList.get(randomGenerator.nextInt(availableMoves.size()));
    }

    public void setSeed(int seed) {
      // Empty use default random
    }

    public Object encode(Integer move) {

      return move;
    }

    public int getSize() {

      return availableMoves.size();
    }

    public Integer noOp() {

      return -1;
    }
  }

  private static final Logger log = LoggerFactory.getLogger(ReinforcementLearningMain.class);

  static final double LEARNING_RATE = 5e-5;
  static final int NET_ITERATIONS = 1;
  static final int TARGET_NET_UPDATE = 100;
  static final int QLEARNING_MAX_STEP = 10000;
  static final int STEPS_EPSILON_GREEDY = 2000;
  static final int BATCH_SIZE = 9;
  static final float MIN_EPSILON = 0.01f;
  static final double ERROR_CLAMP = 0.4;
  static final double GAMMA = 1;
  static final double REWARD_FACTOR = 1;
  static final int MAX_EPOCH_STEP = 9;
  static final int MAX_EXP_REPLAY_SIZE = 5;

  public static void main(String[] args) throws IOException {

    ReinforcementLearningMain reinforcementLearning = new ReinforcementLearningMain();

    DataManager dataManager = new DataManager("output", true);

    MultiLayerNetwork model = reinforcementLearning.createConvolutionalConfiguration();
    TicTacToeGame mdp = new TicTacToeGame();
    mdp.setFetchable(new NeuralNetFetchable<IDQN>() {

      @Override
      public IDQN getNeuralNet() {
        return new DQN(model);
      }
    });

    final MultiLayerNetwork perfectModel =
        ModelSerializer.restoreMultiLayerNetwork("src/main/resources/twoLayerPerfectModel.bin");

    reinforcementLearning.dql = new QLearningDiscreteAdverserial(
        mdp,
        mdp.getFetchable().getNeuralNet(),
        reinforcementLearning.createQLConfiguration(),
        dataManager, perfectModel) {
 
      int lastIterationCount;
      int lastTotalSteps;
      
      @Override
      protected StatEntry trainEpoch() {
        
        StatEntry statEntry = super.trainEpoch();
        reinforcementLearning.statisticEntries.add(statEntry);
        
        return statEntry;
      }

      @Override
      public void postEpoch() {

        int totalGamesPlayed = reinforcementLearning.statisticEntries.size();
        if (totalGamesPlayed > 0 && totalGamesPlayed % TARGET_NET_UPDATE == 0) {

          assert totalGamesPlayed > 0;
          
          int minWins = 0;
          int maxWins = 0;
          int draws = 0;

          for (int lastResultsIndex =
              lastIterationCount; lastResultsIndex < totalGamesPlayed; lastResultsIndex++) {

            StatEntry currentResult =
                reinforcementLearning.statisticEntries.get(lastResultsIndex);
            double result = currentResult.getReward();

            if (currentResult.getStepCounter() == 9) {

              draws++;

            } else if (result < TicTacToeGame.DRAW_REWARD) {

              minWins++;

            } else {

              maxWins++;
            }
          }

          int gamesPlayed = totalGamesPlayed - lastIterationCount;

          String logInfo = "Last iterations (steps): " + gamesPlayed + " (" +
                           (reinforcementLearning.dql.getStepCounter() - lastTotalSteps) + ")\n" +
                           "Last game length avg: " +
                           (double) (reinforcementLearning.dql.getStepCounter() - lastTotalSteps) / gamesPlayed +
                           "\nW=" + minWins + " / L=" + maxWins + " / D=" + draws +
                           " / " + gamesPlayed + " / D Performance: " + ((double) draws /
                                                                         gamesPlayed) +
                           "\nTotal games (steps) = " + totalGamesPlayed + " (" +
                           reinforcementLearning.dql.getStepCounter() + ")";
          log.info(logInfo);

          reinforcementLearning.evaluateNetwork((MultiLayerNetwork) model);

          lastTotalSteps = reinforcementLearning.dql.getStepCounter();
          lastIterationCount = totalGamesPlayed;
        }
      }
    };

    try {
      // start the training
      reinforcementLearning.dql.train();

      int totalGamesPlayed = reinforcementLearning.statisticEntries.size();
      int minWins = 0;
      int maxWins = 0;
      int draws = 0;
      int lastResultsIndex = totalGamesPlayed - 1;
      for (; lastResultsIndex < totalGamesPlayed; lastResultsIndex++) {

        StatEntry currentResult =
            reinforcementLearning.statisticEntries.get(lastResultsIndex);
        double result = currentResult.getReward();

        if (currentResult.getStepCounter() == 9 || result < TicTacToeGame.MAX_WIN_REWARD) {

          draws++;

        } else {

          maxWins++;
        }
      }
      reinforcementLearning.evaluateNetwork((MultiLayerNetwork) model);

      log.info("Last Games (" + draws + " / " + (draws + maxWins + minWins) + "=" + lastResultsIndex + " games): " +
               ((double) draws / (draws + maxWins)));

    } catch (AssertionError ae) {
      log.error(String.valueOf(mdp.currentState.getPlayground()));
      throw ae;
    }
    // good practice
    mdp.close();
  }

  MultiLayerNetwork createConvolutionalConfiguration() {

    ReinforcementLearningMain convolutionalLayerNet = new ReinforcementLearningMain();

    Builder convolutionalLayerNetBuilder =
        convolutionalLayerNet.createConvolutionalConfiguration(createGeneralConfiguration());

    MultiLayerNetwork model = new MultiLayerNetwork(convolutionalLayerNetBuilder.build());
    model.init();

    return model;
  }

  NeuralNetConfiguration.Builder createGeneralConfiguration() {

    return new NeuralNetConfiguration.Builder()
        .seed(DEFAULT_SEED)
        .weightInit(WeightInit.XAVIER)
        .updater(new Adam(LEARNING_RATE));
  }

  QLConfiguration createQLConfiguration() {

    QLConfiguration qlConfiguration = QLConfiguration.builder()
        .maxEpochStep(MAX_EPOCH_STEP) // Max step By epoch
        .maxStep(QLEARNING_MAX_STEP) // Max step
        .expRepMaxSize(MAX_EXP_REPLAY_SIZE) // Max size of experience replay
        .batchSize(BATCH_SIZE) // size of batches
        .targetDqnUpdateFreq(TARGET_NET_UPDATE) // target update (hard)
        .updateStart(0) // num step noop warmup
        .rewardFactor(REWARD_FACTOR) // reward scaling
        .gamma(GAMMA) // gamma
        .errorClamp(ERROR_CLAMP) // td-error clipping
        .minEpsilon(MIN_EPSILON) // min epsilon
        .epsilonNbStep(STEPS_EPSILON_GREEDY) // num step for eps greedy anneal
        .doubleDQN(true)
        .build(); // double DQN

    return qlConfiguration;
  }

  Builder createConvolutionalConfiguration(
      NeuralNetConfiguration.Builder generalConfigBuilder) {

    return new NeuralNetConfiguration.ListBuilder(generalConfigBuilder)
        .layer(0, new DenseLayer.Builder()
            .activation(Activation.TANH)
            .nIn(TicTacToeConstants.COLUMN_NUMBER)
            .nOut(NUMBER_OF_NODES)
            .name(DEFAULT_INPUT_LAYER_NAME)
            .build())
        .layer(1, new DenseLayer.Builder()
            .activation(Activation.TANH)
            .nIn(NUMBER_OF_NODES)
            .nOut(NUMBER_OF_NODES)
            .name(DEFAULT_HIDDEN_LAYER_NAME)
            .build())
        .layer(2, new OutputLayer.Builder()
            .activation(Activation.SOFTMAX)
            .nIn(NUMBER_OF_NODES)
            .nOut(TicTacToeConstants.COLUMN_NUMBER)
            .name(DEFAULT_OUTPUT_LAYER_NAME)
            .build());
  }

  protected void evaluateNetwork(MultiLayerNetwork model) {

    INDArray output = model.output(p1.getFirst());
    Evaluation eval = new Evaluation(COLUMN_NUMBER);
    eval.eval(p1.getSecond(), output);

    if (log.isInfoEnabled()) {
      log.info(eval.stats());
    }

    evaluateOpeningAnswers(model);
  }

  protected void evaluateOpeningAnswers(MultiLayerNetwork convolutionalNetwork) {

    INDArray centerFieldOpeningAnswer = convolutionalNetwork.output(EMPTY_PLAYGROUND.dup().putScalar(FIELD_5, MAX_PLAYER));
    INDArray cornerFieldOpeningAnswer = convolutionalNetwork.output(EMPTY_PLAYGROUND.dup().putScalar(FIELD_9, MAX_PLAYER));

    log.info("Answer to center field opening: {}", centerFieldOpeningAnswer);
    log.info("Answer to last corner field opening: {}", cornerFieldOpeningAnswer);
  }

  protected INDArray generateCenterFieldInputImagesConvolutional() {

    INDArray middleFieldMove = EMPTY_CONVOLUTIONAL_PLAYGROUND.dup();
    INDArray emptyImage1 = Nd4j.ones(1, IMAGE_SIZE, IMAGE_SIZE);
    emptyImage1.putScalar(0, 1, 1, EMPTY_IMAGE_POINT);
    middleFieldMove.putRow(0, emptyImage1);
    middleFieldMove.putScalar(1, 1, 1, OCCUPIED_IMAGE_POINT);
    INDArray graphSingleBatchInput1 = Nd4j.create(1, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE);
    graphSingleBatchInput1.putRow(0, middleFieldMove);
    return graphSingleBatchInput1;
  }

  protected INDArray generateLastCornerFieldInputImagesConvolutional() {

    INDArray cornerFieldMove = EMPTY_CONVOLUTIONAL_PLAYGROUND.dup();
    INDArray emptyImage2 = Nd4j.ones(1, IMAGE_SIZE, IMAGE_SIZE);
    emptyImage2.putScalar(0, 2, 2, EMPTY_IMAGE_POINT);
    cornerFieldMove.putRow(0, emptyImage2);
    cornerFieldMove.putScalar(1, 2, 2, OCCUPIED_IMAGE_POINT);
    INDArray graphSingleBatchInput2 = Nd4j.create(1, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE);
    graphSingleBatchInput2.putRow(0, cornerFieldMove);
    return graphSingleBatchInput2;
  }
}
