package ch.evolutionsoft.rl.alphazero;

import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletionService;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

import org.apache.commons.math3.distribution.EnumeratedIntegerDistribution;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import cc.mallet.types.Dirichlet;
import ch.evolutionsoft.rl.AdversaryLearningConfiguration;
import ch.evolutionsoft.rl.AdversaryLearningConstants;
import ch.evolutionsoft.rl.AdversaryLearningSharedHelper;
import ch.evolutionsoft.rl.AdversaryTrainingExample;
import ch.evolutionsoft.rl.Game;
import ch.evolutionsoft.rl.GraphLoader;

/**
 * {@link AdversaryLearning} is the main class to perform alpha zero learning.
 * It uses game results from 0 to 1 instead of -1 to 1 compared with other
 * implementations.
 * 
 * This affects also the residual net, where a sigmoid activation instead of a
 * tanh is used for the expected value output head.
 * 
 * @author evolutionsoft
 */
@Component
public class AdversaryLearning {

  public static final double DRAW_VALUE = 0.5;
  public static final double DRAW_WEIGHT = 0.5;
  public static final double MAX_WIN = AdversaryLearningConstants.ONE;
  public static final double MIN_WIN = AdversaryLearningConstants.ZERO;

  public static final int NO_MOVE = -2;

  public static final int SEVEN_DIGITS = 7;

  public static final String TEMPMODEL_NAME = "tempmodel.bin";

  private static final Logger log = LoggerFactory.getLogger(AdversaryLearning.class);

  Game initialGame;

  ComputationGraph computationGraph;
  ComputationGraph previousComputationGraph;

  AdversaryLearningConfiguration adversaryLearningConfiguration;

  AdversaryLearningController adversaryLearningController;

  AdversaryLearningSharedHelper sharedHelper;

  int previousNetIteration = -1;

  int iteration = 1;

  boolean restoreTrainingExamples;

  boolean restoreTrainedNeuralNet;

  boolean initialized;

  public AdversaryLearning() {
  }

  public AdversaryLearning(Game game, ComputationGraph computationGraph, AdversaryLearningConfiguration configuration) {

    this.initialize(game, computationGraph, configuration);
  }

  public void initialize(Game game, ComputationGraph computationGraph, AdversaryLearningConfiguration configuration) {

    this.initialGame = game;
    this.computationGraph = computationGraph;
    this.adversaryLearningConfiguration = configuration;
    this.restoreTrainingExamples = configuration.getIterationStart() > 1;
    this.restoreTrainedNeuralNet = configuration.getIterationStart() > 1;

    this.adversaryLearningController = new AdversaryLearningController(this);

    this.sharedHelper = new AdversaryLearningSharedHelper(configuration);

    log.info("Using configuration\n{}", configuration);
  }

  public void performLearning() throws IOException {

    if (this.restoreTrainedNeuralNet) {
      GraphLoader.loadComputationGraph(this.adversaryLearningConfiguration);
      loadTempComputationGraph();

    } else {

      ModelSerializer.writeModel(computationGraph, this.adversaryLearningConfiguration.getBestModelFileName(), true);
    }

    ExecutorService executor = Executors.newSingleThreadExecutor();

    executor.submit(new Callable<Void>() {

      @Override
      public Void call() throws Exception {

        if (AdversaryLearning.this.restoreTrainingExamples) {

          AdversaryLearning.this.sharedHelper.loadEarlierTrainingExamples();
        }

        return null;
      }

    });

    this.iteration = adversaryLearningConfiguration.getIterationStart();

    executor.shutdown();

    try {
      while (!executor.awaitTermination(Long.MAX_VALUE, TimeUnit.SECONDS)) {
        // Wait for monte carlo searches termination
      }
    } catch (InterruptedException ie) {

      Thread.currentThread().interrupt();
      throw new AdversaryLearningRuntimeException(ie);
    }

    this.initialized = true;
  }

  public Set<AdversaryTrainingExample> performIteration() throws IOException {

    if (this.previousNetIteration == this.computationGraph.getIterationCount()) {

      Set<INDArray> trainingBoards = this.sharedHelper.getTrainExampleBoardsByIteration().get(this.iteration);
      Set<AdversaryTrainingExample> previousExamples = new HashSet<>();

      trainingBoards.forEach(tb -> previousExamples.add(this.sharedHelper.getTrainExamplesHistory().get(tb)));

      log.info("returning existing examples from iteration {}", this.iteration);

      return previousExamples;
    }

    final int currentIteration = iteration;
    ThreadPoolExecutor executor = (ThreadPoolExecutor) Executors
        .newFixedThreadPool(adversaryLearningConfiguration.getNumberOfEpisodeThreads());
    CompletionService<List<AdversaryTrainingExample>> completionService = new ExecutorCompletionService<>(executor);

    Set<AdversaryTrainingExample> examplesFromEpisodes = new HashSet<>();

    for (int episode = 1; episode <= adversaryLearningConfiguration
        .getNumberOfEpisodesBeforePotentialUpdate(); episode++) {

      final MonteCarloTreeSearch mcts = new MonteCarloTreeSearch(computationGraph.clone(),
          this.adversaryLearningConfiguration);
      final Game newGameInstance = this.initialGame.createNewInstance();
      completionService.submit(new Callable<List<AdversaryTrainingExample>>() {

        @Override
        public List<AdversaryTrainingExample> call() throws Exception {

          return executeEpisode(currentIteration, newGameInstance, mcts);
        }

      });
    }

    int received = 0;

    executor.shutdown();

    try {
      while (received < adversaryLearningConfiguration.getNumberOfEpisodesBeforePotentialUpdate()) {

        Future<List<AdversaryTrainingExample>> adversaryTrainingExamplesFuture = completionService.take();
        List<AdversaryTrainingExample> currentAdversaryTrainingExamples = adversaryTrainingExamplesFuture.get();
        examplesFromEpisodes.addAll(currentAdversaryTrainingExamples);
        received++;

        log.info("Episode {}-{} ended", currentIteration, received);

        if (log.isDebugEnabled()) {

          log.debug("Got {} potentially new train examples", currentAdversaryTrainingExamples.size());
        }
      }
    } catch (InterruptedException | ExecutionException exception) {

      Thread.currentThread().interrupt();
      throw new AdversaryLearningRuntimeException(exception);
    }

    this.sharedHelper.replaceOldTrainingExamplesWithNewActionProbabilities(examplesFromEpisodes);
    saveTrainExamplesHistory();

    this.iteration++;

    return examplesFromEpisodes;
  }

  public List<AdversaryTrainingExample> executeEpisode(int iteration, Game currentGame, MonteCarloTreeSearch mcts) {

    List<AdversaryTrainingExample> trainExamples = new ArrayList<>();

    int currentPlayer = Game.MAX_PLAYER;

    int moveNumber = 1;
    TreeNode treeNode = new TreeNode(-1, currentGame.getOtherPlayer(currentGame.getCurrentPlayer()), 0, 1.0, 0.5, null);

    while (!currentGame.gameEnded()) {

      INDArray validMoves = currentGame.getValidMoves();
      Set<Integer> validMoveIndices = currentGame.getValidMoveIndices();

      double currentTemperature = adversaryLearningConfiguration.getCurrentTemperature(iteration, moveNumber);
      INDArray actionProbabilities = mcts.getActionValues(currentGame, treeNode, currentTemperature);

      INDArray validActionProbabilities = actionProbabilities.mul(validMoves);
      INDArray normalizedActionProbabilities = validActionProbabilities.div(Nd4j.sum(actionProbabilities));

      List<AdversaryTrainingExample> newTrainingExamples = createNewTrainingExamplesWithSymmetries(iteration,
          currentGame.getCurrentBoard(), currentPlayer, normalizedActionProbabilities);

      trainExamples.removeAll(newTrainingExamples);
      trainExamples.addAll(newTrainingExamples);

      int moveAction = chooseNewMoveAction(validMoveIndices, normalizedActionProbabilities, currentGame);

      currentGame.makeMove(moveAction, currentPlayer);
      moveNumber++;

      treeNode = updateMonteCarloSearchRoot(currentGame, treeNode, moveAction);

      if (currentGame.gameEnded()) {
        handleGameEnded(trainExamples, currentGame, currentPlayer);
      }

      currentPlayer = currentPlayer == Game.MAX_PLAYER ? Game.MIN_PLAYER : Game.MAX_PLAYER;
    }

    return trainExamples;
  }

  public void loadTempComputationGraph() throws IOException {

    String absoluteBestModelPath = AdversaryLearningConfiguration
        .getAbsolutePathFrom(adversaryLearningConfiguration.getBestModelFileName());
    if (!this.adversaryLearningConfiguration.isAlwaysUpdateNeuralNetwork()) {

      this.previousComputationGraph = ModelSerializer.restoreComputationGraph(absoluteBestModelPath, true);
      this.previousComputationGraph.setLearningRate(this.adversaryLearningConfiguration.getLearningRate());
      if (null != this.adversaryLearningConfiguration.getLearningRateSchedule()) {
        this.computationGraph.setLearningRate(this.adversaryLearningConfiguration.getLearningRateSchedule());
      }
      log.info("restored temp model from {}", absoluteBestModelPath);
    }

  }

  boolean updateNeuralNet() throws IOException {

    this.previousNetIteration = this.computationGraph.getIterationCount();
    boolean updateAfterBetterPlayout = false;
    if (!adversaryLearningConfiguration.isAlwaysUpdateNeuralNetwork()) {

      String absoluteTempModelPath = AdversaryLearningConfiguration.getAbsolutePathFrom(TEMPMODEL_NAME);
      ModelSerializer.writeModel(computationGraph, absoluteTempModelPath, true);

      log.info("Write temp model {}", absoluteTempModelPath);

      this.previousComputationGraph = ModelSerializer.restoreComputationGraph(absoluteTempModelPath, true);

      this.computationGraph = GraphLoader.loadComputationGraph(adversaryLearningConfiguration);

      log.info("Challenge new model version with previous model in {} games",
          adversaryLearningConfiguration.getNumberOfGamesToDecideUpdate());

      AdversaryAgentDriver adversaryAgentDriver = new AdversaryAgentDriver(this.previousComputationGraph,
          this.computationGraph);

      int[] gameResults = adversaryAgentDriver.playGames(this.initialGame, adversaryLearningConfiguration);

      log.info("New model wins {} / prev model wins {} / draws {}", gameResults[1], gameResults[0], gameResults[2]);

      double newModelWinDrawRatio = (gameResults[1] + DRAW_WEIGHT * gameResults[2])
          / (gameResults[0] + gameResults[1] + DRAW_WEIGHT * gameResults[2]);
      updateAfterBetterPlayout = newModelWinDrawRatio > adversaryLearningConfiguration
          .getGamesWinRatioThresholdNewNetworkUpdate();

      log.info("New model win/draw ratio against previous model is {} vs configured threshold {}", newModelWinDrawRatio,
          adversaryLearningConfiguration.getGamesWinRatioThresholdNewNetworkUpdate());

      if (!updateAfterBetterPlayout) {

        log.info("Rejecting new model");
        this.computationGraph = ModelSerializer.restoreComputationGraph(absoluteTempModelPath, true);

        log.info("Restored best model from {}", absoluteTempModelPath);
      }

    } else {

      this.computationGraph = GraphLoader.loadComputationGraph(adversaryLearningConfiguration);

      updateAfterBetterPlayout = true;
    }

    if (!adversaryLearningConfiguration.isAlwaysUpdateNeuralNetwork() && updateAfterBetterPlayout) {
      initialGame.evaluateBoardActionExamples(previousComputationGraph);
    }
    initialGame.evaluateBoardActionExamples(computationGraph);
    initialGame.evaluateNetwork(computationGraph);

    createCheckpoint(iteration);

    /**
     * TODO use iteration from NeuralNetUpdater one behind
     */
    log.info("Iteration {} ended", iteration);

    return updateAfterBetterPlayout;
  }

  void createCheckpoint(int iteration) throws IOException {

    StringBuilder prependedZeros = prependZeros(iteration);

    if (0 == iteration % adversaryLearningConfiguration.getCheckPointIterationsFrequency()) {

      String bestModelPath = AdversaryLearningConfiguration
          .getAbsolutePathFrom(adversaryLearningConfiguration.getBestModelFileName());

      String suffix = "";
      String bestModelBasePath = bestModelPath;
      if (adversaryLearningConfiguration.getBestModelFileName().contains(".")) {
        suffix = bestModelPath.substring(bestModelPath.lastIndexOf('.'), bestModelPath.length());
        int suffixLength = suffix.length();
        bestModelBasePath = bestModelPath.substring(0, bestModelPath.length() - suffixLength);
      }
      ModelSerializer.writeModel(computationGraph, bestModelBasePath + prependedZeros + iteration + suffix, true);

      saveTrainExamplesHistory(iteration);
    }
  }

  List<AdversaryTrainingExample> createNewTrainingExamplesWithSymmetries(int iteration, INDArray currentBoard,
      int currentPlayer, INDArray normalizedActionProbabilities) {

    List<AdversaryTrainingExample> newTrainingExamples = new ArrayList<>();

    AdversaryTrainingExample trainingExample = new AdversaryTrainingExample(currentBoard, currentPlayer,
        normalizedActionProbabilities, iteration);

    newTrainingExamples.add(trainingExample);

    List<AdversaryTrainingExample> symmetries = initialGame.getSymmetries(currentBoard.dup(),
        normalizedActionProbabilities.dup(), currentPlayer, iteration);

    Set<AdversaryTrainingExample> addedSymmetries = new HashSet<>();
    addedSymmetries.add(trainingExample);
    for (AdversaryTrainingExample symmetryExample : symmetries) {

      if (!addedSymmetries.contains(symmetryExample)) {
        newTrainingExamples.add(symmetryExample);
        addedSymmetries.add(symmetryExample);
      }
    }

    return newTrainingExamples;
  }

  int chooseNewMoveAction(Set<Integer> validMoveIndices, INDArray normalizedActionProbabilities, Game currentGame) {

    int moveAction;
    if (!hasMoreThanOneMove(validMoveIndices)) {

      moveAction = validMoveIndices.iterator().next();

    } else {

      double alpha = adversaryLearningConfiguration.getDirichletAlpha();
      Dirichlet dirichlet = new Dirichlet(validMoveIndices.size(), alpha);

      INDArray nextDistribution = Nd4j.createFromArray(dirichlet.nextDistribution());
      int[] validIndices = currentGame.getValidIndices(validMoveIndices);

      INDArray reducedValidActionProbabilities = normalizedActionProbabilities.get(Nd4j.createFromArray(validIndices));
      INDArray noiseActionDistribution = reducedValidActionProbabilities
          .mul(1 - adversaryLearningConfiguration.getDirichletWeight())
          .add(nextDistribution.mul(adversaryLearningConfiguration.getDirichletWeight()));

      nextDistribution.close();

      EnumeratedIntegerDistribution distribution = new EnumeratedIntegerDistribution(validIndices,
          noiseActionDistribution.toDoubleVector());

      moveAction = distribution.sample();

      while (!validMoveIndices.contains(moveAction)) {
        // Should not occur with correctly reducedValidActionProbabilities above
        log.warn("Resample invalid random choice move: {} \nvalidIndices = {}\nreducedActionProbs = {}\ngame = \n{}",
            moveAction, validIndices, reducedValidActionProbabilities, currentGame);
        moveAction = distribution.sample();
      }
    }
    return moveAction;
  }

  void handleGameEnded(List<AdversaryTrainingExample> trainExamples, Game currentGame, int currentPlayer) {

    double gameResult = currentGame.getEndResult(currentPlayer);

    if (gameResult != DRAW_VALUE) {

      if (currentPlayer == Game.MIN_PLAYER) {

        gameResult = 1 - gameResult;
      }

      for (AdversaryTrainingExample trainExample : trainExamples) {

        trainExample.setCurrentPlayerValue(
            (float) (trainExample.getCurrentPlayer() == currentPlayer ? gameResult : 1 - gameResult));
      }
    } else {

      for (AdversaryTrainingExample trainExample : trainExamples) {

        trainExample.setCurrentPlayerValue((float) DRAW_VALUE);
      }
    }
  }

  TreeNode updateMonteCarloSearchRoot(Game game, TreeNode lastRoot, int moveAction) {

    if (lastRoot.containsChildMoveIndex(moveAction)) {

      return lastRoot.getChildWithMoveIndex(moveAction);
    } else {

      log.warn("{}", game);
      log.error("no child with move {} " + "found for current root node with last move {}", moveAction,
          lastRoot.lastMove);

      return null;
    }
  }

  void saveTrainExamplesHistory() throws IOException {

    this.sharedHelper.resizeTrainExamplesHistory();

    String trainExamplesByBoardPath = AdversaryLearningConfiguration
        .getAbsolutePathFrom(adversaryLearningConfiguration.getTrainExamplesFileName());
    String suffix = "";
    String trainExamplesBasePath = trainExamplesByBoardPath;
    if (adversaryLearningConfiguration.getTrainExamplesFileName().contains(".")) {
      suffix = trainExamplesByBoardPath.substring(trainExamplesByBoardPath.lastIndexOf('.'),
          trainExamplesByBoardPath.length());
      int suffixLength = suffix.length();
      trainExamplesBasePath = trainExamplesByBoardPath.substring(0, trainExamplesByBoardPath.length() - suffixLength);
    }
    writeMapToFile(trainExamplesByBoardPath,
        trainExamplesBasePath + AdversaryLearningSharedHelper.TRAIN_EXAMPLES_VALUES + suffix,
        this.sharedHelper.getTrainExamplesHistory());
  }

  void saveTrainExamplesHistory(int iteration) throws IOException {

    this.sharedHelper.resizeTrainExamplesHistory();

    StringBuilder prependedZeros = prependZeros(iteration);
    String trainExamplesPath = AdversaryLearningConfiguration
        .getAbsolutePathFrom(adversaryLearningConfiguration.getTrainExamplesFileName());

    String suffix = "";
    String trainExamplesBasePath = trainExamplesPath;
    if (adversaryLearningConfiguration.getTrainExamplesFileName().contains(".")) {
      suffix = trainExamplesPath.substring(trainExamplesPath.lastIndexOf('.'), trainExamplesPath.length());
      int suffixLength = suffix.length();
      trainExamplesBasePath = trainExamplesPath.substring(0, trainExamplesPath.length() - suffixLength);
    }

    String trainExamplesCheckpointBoardsFile = trainExamplesBasePath + prependedZeros + iteration + suffix;
    String trainExamplesCheckpointValuesFile = trainExamplesBasePath
        + AdversaryLearningSharedHelper.TRAIN_EXAMPLES_VALUES + prependedZeros + iteration + suffix;

    this.writeMapToFile(trainExamplesCheckpointBoardsFile, trainExamplesCheckpointValuesFile,
        this.sharedHelper.getTrainExamplesHistory());
  }

  void writeMapToFile(String trainExamplesKeyPath, String trainExamplesValuesPath,
      Map<INDArray, AdversaryTrainingExample> sourceMap) throws IOException {

    if (!sourceMap.isEmpty()) {
      AdversaryTrainingExample example = sourceMap.values().iterator().next();

      long[] boardShape = sourceMap.keySet().iterator().next().shape();
      long[] actionShape = example.getActionIndexProbabilities().shape();

      INDArray allBoardsKey = Nd4j.zeros(sourceMap.size(), boardShape[0], boardShape[1], boardShape[2]);
      INDArray allValues = Nd4j.zeros(sourceMap.size(), actionShape[0] + 3);

      int exampleNumber = 0;
      for (Map.Entry<INDArray, AdversaryTrainingExample> currentExampleEntry : sourceMap.entrySet()) {

        allBoardsKey.putSlice(exampleNumber, currentExampleEntry.getKey());
        INDArray valueNDArray = Nd4j.zeros(actionShape[0] + 3);

        AdversaryTrainingExample value = currentExampleEntry.getValue();
        INDArray actionIndexProbabilities = value.getActionIndexProbabilities();
        for (int actionIndex = 0; actionIndex <= actionShape[0] - 1; actionIndex++) {

          valueNDArray.putScalar(actionIndex, actionIndexProbabilities.getFloat(actionIndex));
        }
        valueNDArray.putScalar(actionShape[0], value.getCurrentPlayer());
        valueNDArray.putScalar(actionShape[0] + 1, value.getCurrentPlayerValue());
        valueNDArray.putScalar(actionShape[0] + 2, value.getIteration());
        allValues.putRow(exampleNumber, valueNDArray);

        exampleNumber++;
      }

      try (DataOutputStream dataOutputStream = new DataOutputStream(new FileOutputStream(trainExamplesKeyPath))) {

        Nd4j.write(allBoardsKey, dataOutputStream);
      }

      try (DataOutputStream dataOutputStream = new DataOutputStream(new FileOutputStream(trainExamplesValuesPath))) {

        Nd4j.write(allValues, dataOutputStream);
      }
    }
  }

  StringBuilder prependZeros(int iteration) {

    int prependingZeros = SEVEN_DIGITS - String.valueOf(iteration).length();

    StringBuilder prependedZeros = new StringBuilder();
    for (int n = 1; n <= prependingZeros; n++) {
      prependedZeros.append('0');
    }
    return prependedZeros;
  }

  boolean hasMoreThanOneMove(Set<Integer> emptyFields) {

    return 1 < emptyFields.size();
  }

  public Map<INDArray, AdversaryTrainingExample> getTrainExamplesHistory() {

    return this.sharedHelper.getTrainExamplesHistory();
  }

  public Map<Integer, Set<INDArray>> getTrainExampleBoardsByIteration() {

    return this.sharedHelper.getTrainExampleBoardsByIteration();
  }
}
