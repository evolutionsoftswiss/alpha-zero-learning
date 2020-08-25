package ch.evolutionsoft.rl4j.tictactoe;

import static ch.evolutionsoft.net.game.tictactoe.TicTacToeConstants.*;
import static ch.evolutionsoft.net.game.NeuralNetConstants.*;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning.QLConfiguration;
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.QLearningDiscrete;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.util.IDataManager.StatEntry;
import org.deeplearning4j.ui.VertxUIServer;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.BaseStatsListener;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.stats.impl.DefaultStatsUpdateConfiguration;
import org.deeplearning4j.ui.stats.impl.SbeStatsReport;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ch.evolutionsoft.net.game.NeuralDataHelper;
import ch.evolutionsoft.net.game.tictactoe.TicTacToeConstants;
import ch.evolutionsoft.net.game.tictactoe.TicTacToeNeuralDataConverter;

public class ReinforcementLearningMain {

  Pair<INDArray, INDArray> p1 = TicTacToeNeuralDataConverter.stackConvolutionalPlaygroundLabels(
      TicTacToeNeuralDataConverter.convertMiniMaxPlaygroundLabelsToConvolutionalData(
          NeuralDataHelper.readAll("/inputsMin.txt", "/labelsMin.txt")));

  QLearningDiscrete<TicTacToeState> dql;

  List<StatEntry> statisticEntries = new LinkedList<>();

  private static final Logger log = LoggerFactory.getLogger(ReinforcementLearningMain.class);

  static final double LEARNING_RATE = 5e-6;
  static final int TARGET_NET_UPDATE = 500;
  static final int QLEARNING_MAX_STEP = 200000;
  static final int STEPS_EPSILON_GREEDY = 500;
  static final int BATCH_SIZE = 512;
  static final float MAX_EPSILON = 1f;
  static final float MIN_EPSILON = 0.2f;
  static final double ERROR_CLAMP = 0.1;
  static final double GAMMA = 0.95;
  static final double REWARD_FACTOR = 9;
  static final int MAX_EPOCH_STEP = 9;
  static final int MAX_EXP_REPLAY_SIZE = 9;
  static final int NUM_STEPS_NO_UPDATE = 0;
  
  public static final INDArray ZEROS_PLAYGROUND_IMAGE = Nd4j.zeros(1, IMAGE_SIZE, IMAGE_SIZE);
  public static final INDArray ONES_PLAYGROUND_IMAGE = Nd4j.ones(1, IMAGE_SIZE, IMAGE_SIZE);
  
  public static final INDArray EMPTY_CONVOLUTIONAL_PLAYGROUND = Nd4j.create(IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE);
  static {
    EMPTY_CONVOLUTIONAL_PLAYGROUND.putRow(EMPTY_FIELDS_CHANNEL, ONES_PLAYGROUND_IMAGE);
    EMPTY_CONVOLUTIONAL_PLAYGROUND.putRow(MAX_PLAYER_CHANNEL, ZEROS_PLAYGROUND_IMAGE);
    EMPTY_CONVOLUTIONAL_PLAYGROUND.putRow(MIN_PLAYER_CHANNEL, ZEROS_PLAYGROUND_IMAGE);
  }

  static class TicTacToeState implements Encodable {

    private INDArray playground = EMPTY_CONVOLUTIONAL_PLAYGROUND;

    int depth = 0;

    public TicTacToeState(INDArray playground, int depth) {

      this.playground = playground;
      this.depth = depth;
    }

    public INDArray getPlayground() {
      return playground.dup();
    }

    public TicTacToeState makeMove(int flatIndex) {

      int row = flatIndex / IMAGE_CHANNELS;
      int column = flatIndex % IMAGE_CHANNELS;
      
      INDArray newPlayground = this.getPlayground();
      newPlayground.putScalar(EMPTY_FIELDS_CHANNEL, row, column, EMPTY_FIELD_VALUE);
      newPlayground.putScalar(getCurrentPlayer(), row, column, OCCUPIED_IMAGE_POINT);

      return new TicTacToeState(newPlayground, ++this.depth);
    }

    public int getCurrentPlayer() {

      if (depth % 2 == 0) {

        return MAX_PLAYER_CHANNEL;
      }

      return MIN_PLAYER_CHANNEL;
    }

    @Override
    public double[] toArray() {

      return getPlayground().reshape(1, (long) IMAGE_CHANNELS * IMAGE_SIZE * IMAGE_SIZE).toDoubleVector();
    }
  }

  static class TicTacToeAction extends DiscreteSpace {

    Set<Integer> availableMoves;

    public TicTacToeAction(Set<Integer> availableMoves) {

      super(availableMoves.size());
      this.availableMoves = new HashSet<>(availableMoves);
    }

    @Override
    public Integer randomAction() {

      List<Integer> emptyIndicesList = new ArrayList<>(availableMoves);
      return emptyIndicesList.get(randomGenerator.nextInt(availableMoves.size()));
    }

    public void setSeed(int seed) {
      // Empty use default random
    }

    @Override
    public Object encode(Integer move) {

      return move;
    }

    @Override
    public int getSize() {

      return availableMoves.size();
    }

    @Override
    public Integer noOp() {

      return -1;
    }
  }

  public static void main(String[] args) throws IOException {

    ReinforcementLearningMain reinforcementLearning = new ReinforcementLearningMain();

    ComputationGraph model = reinforcementLearning.createConvolutionalConfiguration();
    //ComputationGraph model = ModelSerializer.restoreComputationGraph("src/main/resources/rl-model-2.bin");
    
    if (log.isInfoEnabled()) {
      log.info(model.summary());
    }
    
    TicTacToeGame mdp = new TicTacToeGame();
    mdp.setFetchable(() -> new ConvolutionalNeuralNetDQN(model));

    setupDql(reinforcementLearning, model, mdp);

    performTraining(reinforcementLearning, model, mdp);
    
    ModelSerializer.writeModel(model, "src/main/resources/rl-model-1.bin", true);
    
    // good practice
    mdp.close();
  }

  protected static void setupDql(ReinforcementLearningMain reinforcementLearning, ComputationGraph model,
      TicTacToeGame mdp) throws IOException {

    VertxUIServer uiServer = (VertxUIServer) UIServer.getInstance();
 
    StatsStorage statsStorage = new InMemoryStatsStorage();
    uiServer.attach(statsStorage);
    try {
      uiServer.start();
    } catch (Exception e) {
      log.error("Error starting UI server", e);
    }
    
    StatsListener statsListener = new StatsListener(statsStorage, 1);
    
    statsListener.setUpdateConfig(new DefaultStatsUpdateConfiguration.Builder().
        collectHistogramsActivations(false).
        collectHistogramsGradients(false).
        collectHistogramsParameters(false).
        collectHistogramsUpdates(false).
        collectLearningRates(true).
        collectMeanActivations(true).
        collectMeanGradients(true).
        collectMeanMagnitudesActivations(false).
        collectMeanMagnitudesGradients(false).
        collectMeanMagnitudesParameters(false).
        collectMeanMagnitudesUpdates(false).
        collectMeanParameters(true).
        collectMeanUpdates(true).
        collectStdevActivations(true).
        collectStdevGradients(true).
        collectStdevParameters(true).
        collectStdevUpdates(true).build());
    model.addListeners(statsListener);

    final ComputationGraph perfectModel =
        ModelSerializer.restoreComputationGraph("src/main/resources/TicTacToeResNet.bin");
 
    reinforcementLearning.dql = new QLearningDiscreteAdverserial(
        mdp,
        mdp.getFetchable().getNeuralNet(),
        reinforcementLearning.createQLConfiguration(),
        perfectModel) {
 
      int lastIterationCount;
      int lastTotalSteps;
      
      @Override
      protected StatEntry trainEpoch() {
        
        StatEntry statEntry = super.trainEpoch();
        
        reinforcementLearning.statisticEntries.add(statEntry);

        int totalGamesPlayed = reinforcementLearning.statisticEntries.size();
        if (totalGamesPlayed % TARGET_NET_UPDATE == 0) {
          
          int minWins = 0;
          int maxWins = 0;
          int draws = 0;

          for (int lastResultsIndex =
              lastIterationCount; lastResultsIndex < totalGamesPlayed; lastResultsIndex++) {

            QLStatEntry currentResult =
                (QLStatEntry) reinforcementLearning.statisticEntries.get(lastResultsIndex);
            double result = currentResult.getReward();
            int episodeLength = currentResult.getEpisodeLength();

            if (episodeLength >= COLUMN_COUNT) {

              draws++; 
            }

            // dependent from calculateReward
            else if (result / episodeLength > 0.9) {

              maxWins++;

            } else {

              minWins++;

            }
          }

          int gamesPlayed = totalGamesPlayed - lastIterationCount;

          double drawFactor = (double) draws / gamesPlayed;
          
          String logInfo = "Last iterations (steps): " + gamesPlayed + " (" +
                           (reinforcementLearning.dql.getStepCounter() - lastTotalSteps) + ")\n" +
                           "Last game length avg: " +
                           (double) (reinforcementLearning.dql.getStepCounter() - lastTotalSteps) / gamesPlayed +
                           "\nL=" + minWins + " / W=" + maxWins + " / D=" + draws +
                           " / " + gamesPlayed + " / D Performance: " + drawFactor +
                           "\nTotal games (steps) = " + totalGamesPlayed + " (" +
                           reinforcementLearning.dql.getStepCounter() + ")";
          log.info(logInfo);

          reinforcementLearning.evaluateNetwork(model);
          
          final SbeStatsReport drawStatsReport = new SbeStatsReport();
          drawStatsReport.setWorkerID(statsListener.getWorkerID());
          drawStatsReport.setSessionID(statsListener.getSessionID());
          drawStatsReport.setTypeID(BaseStatsListener.TYPE_ID);
          drawStatsReport.setTimeStamp(System.currentTimeMillis());
          drawStatsReport.reportIterationCount(lastTotalSteps);
          drawStatsReport.reportScore(model.score());
          statsStorage.putUpdate(drawStatsReport);
          statsListener.getStorageRouter().putUpdate(drawStatsReport);

          lastTotalSteps = reinforcementLearning.dql.getStepCounter();
          lastIterationCount = totalGamesPlayed;
          
          statsListener.iterationDone(model, lastTotalSteps, totalGamesPlayed);
         
          mdp.switchTrainingPlayer();
          log.info("Training player switched to {}", mdp.getTrainingPlayer());
        }

        return statEntry;
      }
    };
  }

  // Integer division is done by none zero values here
  @SuppressWarnings("java:S3518")
  protected static void performTraining(ReinforcementLearningMain reinforcementLearning, ComputationGraph model,
      TicTacToeGame mdp) throws AssertionError {
    try {
      
      // start the training
      reinforcementLearning.dql.train();

      int totalGamesPlayed = reinforcementLearning.statisticEntries.size();
      int minWins = 0;
      int maxWins = 0;
      int draws = 0;
      int lastResultsIndex = 0;

      for (; lastResultsIndex < totalGamesPlayed; lastResultsIndex++) {

        StatEntry currentResult =
            reinforcementLearning.statisticEntries.get(lastResultsIndex);

        if (currentResult.getStepCounter() == 9) {

          draws++;

        } else {

          maxWins++;
        }
      }
      reinforcementLearning.evaluateNetwork(model);

      log.info("Last Games ({} / {} = {} games): {}", draws, (draws + maxWins + minWins), lastResultsIndex, (double) draws / (draws + maxWins));
      
    } catch (AssertionError ae) {
      log.error(String.valueOf(mdp.currentState.getPlayground()));
      throw ae;
    }
  }

  ComputationGraph createConvolutionalConfiguration() {

    ConvolutionResidualNet convolutionalLayerNet = new ConvolutionResidualNet(LEARNING_RATE);

    ComputationGraphConfiguration convolutionalLayerNetConfiguration =
        convolutionalLayerNet.createConvolutionalGraphConfiguration();

    ComputationGraph net = new ComputationGraph(convolutionalLayerNetConfiguration);
    net.init();

    return net;
  }

  NeuralNetConfiguration.Builder createGeneralConfiguration() {

    return new NeuralNetConfiguration.Builder()
        .seed(DEFAULT_SEED)
        .weightInit(WeightInit.XAVIER)
        .updater(new RmsProp(LEARNING_RATE));
  }

  QLConfiguration createQLConfiguration() {

    return QLConfiguration.builder()
        .seed(DEFAULT_SEED)
        .maxEpochStep(MAX_EPOCH_STEP) // Max step By epoch
        .maxStep(QLEARNING_MAX_STEP) // Max step
        .expRepMaxSize(MAX_EXP_REPLAY_SIZE) // Max size of experience replay
        .batchSize(BATCH_SIZE) // size of batches
        .targetDqnUpdateFreq(TARGET_NET_UPDATE) // target update (hard)
        .updateStart(NUM_STEPS_NO_UPDATE) // num step noop warmup
        .rewardFactor(REWARD_FACTOR) // reward scaling
        .gamma(GAMMA) // gamma
        .errorClamp(ERROR_CLAMP) // td-error clipping
        .minEpsilon(MIN_EPSILON) // min epsilon
        .epsilonNbStep(STEPS_EPSILON_GREEDY) // num step for eps greedy anneal
        .doubleDQN(true) // double DQN
        .build(); 
  }

  protected void evaluateNetwork(MultiLayerNetwork model) {

    INDArray output = model.output(p1.getFirst());
    Evaluation eval = new Evaluation(TicTacToeConstants.COLUMN_COUNT);
    eval.eval(p1.getSecond(), output);

    if (log.isInfoEnabled()) {
      log.info(eval.stats());
    }

    evaluateOpeningAnswers(model);
  }

  protected void evaluateNetwork(ComputationGraph model) {

    INDArray output = model.output(p1.getFirst())[0];
    Evaluation eval = new Evaluation(TicTacToeConstants.COLUMN_COUNT);
    eval.eval(p1.getSecond(), output);

    if (log.isInfoEnabled()) {
      log.info(eval.stats());
    }

    evaluateOpeningAnswers(model);
  }

  protected void evaluateOpeningAnswers(MultiLayerNetwork convolutionalNetwork) {

    INDArray centerFieldOpeningAnswer =
        convolutionalNetwork.output(EMPTY_PLAYGROUND.dup().putScalar(FIELD_5, MAX_PLAYER));
    INDArray cornerFieldOpeningAnswer = convolutionalNetwork.output(
        EMPTY_PLAYGROUND.dup().putScalar(FIELD_9, MAX_PLAYER));
    INDArray fieldSixCenterOpeningAnswer = convolutionalNetwork.output(
        EMPTY_PLAYGROUND.dup().
        putScalar(FIELD_6, MAX_PLAYER).
        putScalar(FIELD_5, MIN_PLAYER));

    log.info("Answer to center field opening: {}", centerFieldOpeningAnswer);
    log.info("Answer to last corner field opening: {}", cornerFieldOpeningAnswer);
    log.info("Answer to field six and center response opening: {}", fieldSixCenterOpeningAnswer);
  }

  protected void evaluateOpeningAnswers(ComputationGraph convolutionalNetwork) {

    INDArray centerFieldOpeningAnswer =
        convolutionalNetwork.output(
            generateCenterFieldInputImagesConvolutional())[0];
    INDArray cornerFieldOpeningAnswer = convolutionalNetwork.output(
        generateLastCornerFieldInputImagesConvolutional())[0];
    INDArray fieldSixCenterOpeningAnswer = convolutionalNetwork.output(
        generateFieldSixAndCenterInputImagesConvolutional())[0];

    log.info("Answer to center field opening: {}", centerFieldOpeningAnswer);
    log.info("Answer to last corner field opening: {}", cornerFieldOpeningAnswer);
    log.info("Answer to field six and center response opening: {}", fieldSixCenterOpeningAnswer);
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

  protected INDArray generateFieldSixAndCenterInputImagesConvolutional() {

    INDArray fieldSixMaxCenterMinMove = EMPTY_CONVOLUTIONAL_PLAYGROUND.dup();
    INDArray emptyImage1 = Nd4j.ones(1, IMAGE_SIZE, IMAGE_SIZE);
    INDArray emptyImage2 = Nd4j.ones(1, IMAGE_SIZE, IMAGE_SIZE);
    emptyImage2.putScalar(0, 1, 2, EMPTY_IMAGE_POINT);
    emptyImage2.putScalar(0, 1, 1, EMPTY_IMAGE_POINT);
    fieldSixMaxCenterMinMove.putRow(0, emptyImage1);
    fieldSixMaxCenterMinMove.putScalar(1, 1, 2, OCCUPIED_IMAGE_POINT);
    fieldSixMaxCenterMinMove.putRow(0, emptyImage2);
    fieldSixMaxCenterMinMove.putScalar(2, 1, 1, OCCUPIED_IMAGE_POINT);
    INDArray graphSingleBatchInput2 = Nd4j.create(1, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE);
    graphSingleBatchInput2.putRow(0, fieldSixMaxCenterMinMove);
    return graphSingleBatchInput2;
  }
}
