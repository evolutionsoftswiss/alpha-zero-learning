package ch.evolutionsoft.rl;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SortedSet;
import java.util.TreeSet;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletionService;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.stream.Collectors;

import org.apache.commons.math3.distribution.EnumeratedIntegerDistribution;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import cc.mallet.types.Dirichlet;

/**
 * {@link AdversaryLearning} is the main class to perform alpha zero learning.
 * It uses game results from 0 to 1 instead of -1 to 1 compared with other implementations.
 * 
 * This affects also the residual net, where a sigmoid activation instead of a tanh is
 * used for the expected value output head.
 * 
 * @author evolutionsoft
 */
public class AdversaryLearning {

  public static final double DRAW_VALUE = 0.5;
  public static final double DRAW_WEIGHT = 0.5;
  public static final double MAX_WIN = AdversaryLearningConstants.ONE;
  public static final double MIN_WIN = AdversaryLearningConstants.ZERO;

  public static final int NO_MOVE = -2;
  
  public static final int SEVEN_DIGITS = 7;
  
  public static final String TEMPMODEL_NAME = "tempmodel.bin";
  public static final String TRAIN_EXAMPLES_VALUES = "Values";

  private static final Logger log = LoggerFactory.getLogger(AdversaryLearning.class);
  
  Map<INDArray, AdversaryTrainingExample> trainExamplesHistory = new HashMap<>();

  Map<Integer, Set<INDArray>> trainExampleBoardsByIteration = new HashMap<>();

  Game initialGame;

  ComputationGraph computationGraph;
  ComputationGraph previousComputationGraph;

  AdversaryLearningConfiguration adversaryLearningConfiguration;
  
  AdversaryLearningController adversaryLearningController;
  
  int iteration;

  boolean restoreTrainingExamples;

  boolean restoreTrainedNeuralNet;

  public AdversaryLearning(Game game, ComputationGraph computationGraph, AdversaryLearningConfiguration configuration) {

    this.initialGame = game;
    this.computationGraph = computationGraph;
    this.adversaryLearningConfiguration = configuration;
    this.restoreTrainingExamples = configuration.getIterationStart() > 1;
    this.restoreTrainedNeuralNet = configuration.getIterationStart() > 1;

    this.adversaryLearningController = new AdversaryLearningController(this);
    
    log.info("Using configuration\n{}", configuration);
  }

  public void performLearning() throws IOException {

    if (this.restoreTrainedNeuralNet) {
      GraphLoader.loadComputationGraph(this.adversaryLearningConfiguration);
      loadTempComputationGraph();
    
    } else {
      
      ModelSerializer.writeModel(
          computationGraph,
          this.adversaryLearningConfiguration.getBestModelFileName(),
          true);
    }
    
    loadEarlierTrainingExamples();

    this.iteration = adversaryLearningConfiguration.getIterationStart();
    
  }

  public List<AdversaryTrainingExample> performIteration() throws IOException {

    this.trainExampleBoardsByIteration.put(iteration, new HashSet<>());
    final int currentIteration = iteration;
    ThreadPoolExecutor executor = (ThreadPoolExecutor) Executors.newFixedThreadPool(
        adversaryLearningConfiguration.getNumberOfEpisodeThreads());
    CompletionService<List<AdversaryTrainingExample>> completionService = new ExecutorCompletionService<>(executor);
    
    Set<AdversaryTrainingExample> examplesFromEpisodes = new HashSet<>();
    
    for (int episode = 1; episode <= adversaryLearningConfiguration.getNumberOfEpisodesBeforePotentialUpdate(); episode++) {

      final MonteCarloTreeSearch mcts = new MonteCarloTreeSearch(
          computationGraph.clone(),
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
      while(received < adversaryLearningConfiguration.getNumberOfEpisodesBeforePotentialUpdate()) {
 
        Future<List<AdversaryTrainingExample>> adversaryTrainingExamplesFuture = completionService.take();
        List<AdversaryTrainingExample> currentAdversaryTrainingExamples = adversaryTrainingExamplesFuture.get();
        examplesFromEpisodes.addAll(currentAdversaryTrainingExamples);
        received++;
        
        log.info("Episode {}-{} ended", iteration, received);

        if (log.isDebugEnabled()) {
          
          log.debug("Got {} potentially new train examples", currentAdversaryTrainingExamples.size());
        }
      }
    } catch (InterruptedException | ExecutionException exception) {
     
      Thread.currentThread().interrupt();
      throw new AdversaryLearningRuntimeException(exception);
    }

    replaceOldTrainingExamplesWithNewActionProbabilities(examplesFromEpisodes);
 
    saveTrainExamplesHistory();

    boolean updateAfterBetterPlayout = updateNeuralNet();

    if (adversaryLearningConfiguration.isAlwaysUpdateNeuralNetwork() || 
        updateAfterBetterPlayout) {

      log.info("Accepting new model");
      String absoluteBestModelPath =
          adversaryLearningConfiguration.getAbsolutePathFrom(adversaryLearningConfiguration.getBestModelFileName());
      ModelSerializer.writeModel(computationGraph,
          absoluteBestModelPath,
          true);
      
      log.info("Write new model {}", absoluteBestModelPath);

      if (updateAfterBetterPlayout) {
        initialGame.evaluateBoardActionExamples(previousComputationGraph);
      }
      initialGame.evaluateBoardActionExamples(computationGraph);
      initialGame.evaluateNetwork(computationGraph);
    }

    createCheckpoint(iteration);


    List<AdversaryTrainingExample> shuffledTrainExamples = new ArrayList<>(this.trainExamplesHistory.values());
    Collections.shuffle(shuffledTrainExamples);
    
    log.info("Iteration {} ended", iteration);

    this.iteration++;

    return shuffledTrainExamples;
  }

  public List<AdversaryTrainingExample> executeEpisode(
      int iteration,
      Game currentGame,
      MonteCarloTreeSearch mcts) {

    List<AdversaryTrainingExample> trainExamples = new ArrayList<>();

    int currentPlayer = Game.MAX_PLAYER;

    int moveNumber = 1;
    TreeNode treeNode = new TreeNode(-1, currentGame.getOtherPlayer(currentGame.currentPlayer), 0, 1.0, 0.5, null);

    while (!currentGame.gameEnded()) {

      INDArray validMoves = currentGame.getValidMoves();
      Set<Integer> validMoveIndices = currentGame.getValidMoveIndices();

      double currentTemperature = adversaryLearningConfiguration.getCurrentTemperature(iteration, moveNumber);
      INDArray actionProbabilities = mcts.getActionValues(currentGame, treeNode, currentTemperature);

      INDArray validActionProbabilities = actionProbabilities.mul(validMoves);
      INDArray normalizedActionProbabilities = validActionProbabilities.div(Nd4j.sum(actionProbabilities));

      List<AdversaryTrainingExample> newTrainingExamples = 
          createNewTrainingExamplesWithSymmetries(iteration, currentGame.getCurrentBoard(), currentPlayer,
              normalizedActionProbabilities);

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

    String absoluteBestModelPath =
        adversaryLearningConfiguration.getAbsolutePathFrom(adversaryLearningConfiguration.getBestModelFileName());
    if (!this.adversaryLearningConfiguration.isAlwaysUpdateNeuralNetwork()) {

      this.previousComputationGraph = ModelSerializer.restoreComputationGraph(absoluteBestModelPath, true);
      this.previousComputationGraph.setLearningRate(this.adversaryLearningConfiguration.getLearningRate());
      if (null != this.adversaryLearningConfiguration.getLearningRateSchedule()) {
        this.computationGraph.setLearningRate(this.adversaryLearningConfiguration.getLearningRateSchedule());
      }
      log.info("restored temp model from {}", absoluteBestModelPath);
    }
    
  }

  public void loadEarlierTrainingExamples() throws IOException {

    if (restoreTrainingExamples) {

      log.info("Restoring trainExamplesByBoard history map, this may take a while...");
      
      String trainExamplesFile = adversaryLearningConfiguration.getTrainExamplesFileName();
      loadMapFromFile(trainExamplesFile, this.trainExamplesHistory);

      log.info("Restoring exampleBoardsByIteration from trainExamplesByBoard map...");
      
      this.trainExampleBoardsByIteration = this.trainExamplesHistory.values().stream().collect(
          Collectors.groupingBy(AdversaryTrainingExample::getIteration,
          Collectors.mapping(AdversaryTrainingExample::getBoard, Collectors.toSet())));

      log.info("Train examples maps restored from {}", trainExamplesFile);
      log.info("trainExamplesByBoard map has {} restored AdversaryTrainingExamples entries", this.trainExamplesHistory.size());
      log.info("exampleBoardsByIteration map has {} restored Set of boards entries with total {} examples", this.trainExampleBoardsByIteration.size(), this.countAllExampleBoardsByIteration());
    }
  }

  void loadMapFromFile(String trainExamplesFile, Map<INDArray, AdversaryTrainingExample> targetMap) throws IOException {
    
    String suffix = "";
    String trainExamplesBasePath = trainExamplesFile;
    if (adversaryLearningConfiguration.getTrainExamplesFileName().contains(".")) {
      suffix = trainExamplesFile.substring(trainExamplesFile.lastIndexOf('.'), trainExamplesFile.length());
      int suffixLength = suffix.length();
      trainExamplesBasePath = trainExamplesFile.substring(0, trainExamplesFile.length() - suffixLength);
    }
    INDArray storedBoardKeys;
    try (DataInputStream dataInputStream =
        new DataInputStream(new FileInputStream(trainExamplesFile))) {
      storedBoardKeys = Nd4j.read(dataInputStream);
    }
    INDArray storedValues;
    try (DataInputStream dataInputStream =
        new DataInputStream(new FileInputStream(trainExamplesBasePath + TRAIN_EXAMPLES_VALUES + suffix))) {
      storedValues =  Nd4j.read(dataInputStream);
    }

    long[] actionShape = storedValues.shape();
    int actionIndicesCount = (int) (actionShape[1] - 3);
    for (int index = 0; index < storedBoardKeys.shape() [0]; index++) {
      
      INDArray currentBoardKey = storedBoardKeys.slice(index);
      INDArray currentStoredValue = storedValues.getRow(index);
      INDArray actionIndexProbs = Nd4j.zeros(actionIndicesCount);
      
      for (int actionIndex = 0; actionIndex < actionIndicesCount; actionIndex++) {
        
        actionIndexProbs.putScalar(actionIndex, currentStoredValue.getFloat(actionIndex));
      }
      int player = currentStoredValue.getInt(actionIndicesCount);
      float playerValue = currentStoredValue.getFloat(actionIndicesCount + 1L);
      int iterationValue = currentStoredValue.getInt(actionIndicesCount + 2);
      AdversaryTrainingExample currentAdversaryExample =
          new AdversaryTrainingExample(currentBoardKey, player, actionIndexProbs, iterationValue);
      currentAdversaryExample.setCurrentPlayerValue(playerValue);
      
      targetMap.put(currentBoardKey, currentAdversaryExample);
    }
      
    int size = targetMap.size();
    log.info("Restored train examples from {} with {} train examples",
         trainExamplesFile,
         size);
  }

  void replaceOldTrainingExamplesWithNewActionProbabilities(Collection<AdversaryTrainingExample> newExamples) {

    int replacedNumber = 0;
    Set<INDArray> newIterationBoards = new HashSet<>();
    int currentIteration = newExamples.iterator().next().getIteration();

    for (AdversaryTrainingExample currentExample : newExamples) {
      
      INDArray currentBoard = currentExample.getBoard();
      newIterationBoards.add(currentBoard);
      AdversaryTrainingExample oldExample = this.trainExamplesHistory.put(currentBoard, currentExample);
      
      if (null != oldExample && oldExample.getIteration() != currentIteration) {
        Set<INDArray> boardEntriesByOldIteration = this.trainExampleBoardsByIteration.get(oldExample.getIteration());
        boardEntriesByOldIteration.remove(currentBoard);
        
        if (log.isDebugEnabled()) {
          replacedNumber++;
        }
      }
    }

    Set<INDArray> previousBordsByIteration = this.trainExampleBoardsByIteration.get(currentIteration);
    previousBordsByIteration.addAll(newIterationBoards);
    
    if (log.isDebugEnabled()) {
      
      int listTotalSize = countAllExampleBoardsByIteration();
      log.debug("Updated {} examples with same board from earlier iterations, remaining {} examples may are new",
          replacedNumber,
          newExamples.size() - replacedNumber);
      log.debug("New trainExamplesByBoard history map and exampleBoardsByIteration history map size {} and {}",
          this.trainExamplesHistory.size(),
          listTotalSize);
    }
  }

  int countAllExampleBoardsByIteration() {

    int listTotalSize = 0;
    for (Set<INDArray> current : this.trainExampleBoardsByIteration.values()) {
      listTotalSize += current.size();
    }
    return listTotalSize;
  }

  boolean updateNeuralNet() throws IOException {
    
    boolean updateAfterBetterPlayout = false;
    if (!adversaryLearningConfiguration.isAlwaysUpdateNeuralNetwork()) {

      String absoluteTempModelPath = adversaryLearningConfiguration.getAbsolutePathFrom(TEMPMODEL_NAME);
      ModelSerializer.writeModel(computationGraph, absoluteTempModelPath, true);
      
      log.info("Write temp model {}", absoluteTempModelPath);
      
      this.previousComputationGraph = ModelSerializer.restoreComputationGraph(absoluteTempModelPath, true);

      this.computationGraph = GraphLoader.loadComputationGraph(adversaryLearningConfiguration);

      log.info("Challenge new model version with previous model in {} games", adversaryLearningConfiguration.getNumberOfGamesToDecideUpdate());
      
      AdversaryAgentDriver adversaryAgentDriver = new AdversaryAgentDriver(
          this.previousComputationGraph,
          this.computationGraph);

      int[] gameResults = adversaryAgentDriver.playGames(this.initialGame, adversaryLearningConfiguration);

      log.info("New model wins {} / prev model wins {} / draws {}", gameResults[1], gameResults[0], gameResults[2]);

      double newModelWinDrawRatio = (gameResults[1] + DRAW_WEIGHT * gameResults[2])
          / (gameResults[0] + gameResults[1] + DRAW_WEIGHT * gameResults[2]);
      updateAfterBetterPlayout = newModelWinDrawRatio > adversaryLearningConfiguration
              .getGamesWinRatioThresholdNewNetworkUpdate();

      log.info("New model win/draw ratio against previous model is {} vs configured threshold {}",
          newModelWinDrawRatio,
          adversaryLearningConfiguration.getGamesWinRatioThresholdNewNetworkUpdate());
      
      if (!updateAfterBetterPlayout) {

        log.info("Rejecting new model");
        this.computationGraph = ModelSerializer.restoreComputationGraph(absoluteTempModelPath, true);
        
        log.info("Restored best model from {}", absoluteTempModelPath);
      }

    } else {

      this.computationGraph = GraphLoader.loadComputationGraph(adversaryLearningConfiguration);
    }

    return updateAfterBetterPlayout;
  }

  void createCheckpoint(int iteration) throws IOException {

    StringBuilder prependedZeros = prependZeros(iteration);
    
    if (0 == iteration % adversaryLearningConfiguration.getCheckPointIterationsFrequency()) {

      String bestModelPath = adversaryLearningConfiguration.getAbsolutePathFrom(adversaryLearningConfiguration.getBestModelFileName());

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

  List<AdversaryTrainingExample> createNewTrainingExamplesWithSymmetries(int iteration,
      INDArray currentBoard, int currentPlayer, INDArray normalizedActionProbabilities) {

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
        log.warn("Resample invalid random choice move: {} \nvalidIndices = {}\nreducedActionProbs = {}\ngame = \n{}", moveAction,
            validIndices, reducedValidActionProbabilities, currentGame);
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
      }
      else {

        log.warn("{}", game);
        log.error("no child with move {} " +
                "found for current root node with last move {}", moveAction, lastRoot.lastMove);
        
        return null;
      }
  }

  void saveTrainExamplesHistory() throws IOException {

    this.resizeTrainExamplesHistory();

    String trainExamplesByBoardPath = adversaryLearningConfiguration.getAbsolutePathFrom(
        adversaryLearningConfiguration.getTrainExamplesFileName());
    String suffix = "";
    String trainExamplesBasePath = trainExamplesByBoardPath;
    if (adversaryLearningConfiguration.getTrainExamplesFileName().contains(".")) {
      suffix = trainExamplesByBoardPath.substring(trainExamplesByBoardPath.lastIndexOf('.'), trainExamplesByBoardPath.length());
      int suffixLength = suffix.length();
      trainExamplesBasePath = trainExamplesByBoardPath.substring(0, trainExamplesByBoardPath.length() - suffixLength);
    }
    writeMapToFile(trainExamplesByBoardPath, trainExamplesBasePath + TRAIN_EXAMPLES_VALUES + suffix, this.trainExamplesHistory);
  }

  void saveTrainExamplesHistory(int iteration) throws IOException {

    this.resizeTrainExamplesHistory();

    StringBuilder prependedZeros = prependZeros(iteration);
    String trainExamplesPath = adversaryLearningConfiguration.getAbsolutePathFrom(
        adversaryLearningConfiguration.getTrainExamplesFileName());
 
    String suffix = "";
    String trainExamplesBasePath = trainExamplesPath;
    if (adversaryLearningConfiguration.getTrainExamplesFileName().contains(".")) {
      suffix = trainExamplesPath.substring(trainExamplesPath.lastIndexOf('.'), trainExamplesPath.length());
      int suffixLength = suffix.length();
      trainExamplesBasePath = trainExamplesPath.substring(0, trainExamplesPath.length() - suffixLength);
    }
    
    String trainExamplesCheckpointBoardsFile = trainExamplesBasePath + 
        prependedZeros + iteration + suffix;
    String trainExamplesCheckpointValuesFile = trainExamplesBasePath + TRAIN_EXAMPLES_VALUES +
        prependedZeros + iteration + suffix;

    this.writeMapToFile(trainExamplesCheckpointBoardsFile, trainExamplesCheckpointValuesFile, this.trainExamplesHistory);
  }

  void writeMapToFile(String trainExamplesKeyPath, String trainExamplesValuesPath, Map<INDArray, AdversaryTrainingExample> sourceMap) throws IOException {

    AdversaryTrainingExample example = sourceMap.values().iterator().next();
    
    long[] boardShape = example.getBoard().shape();
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

    try (DataOutputStream dataOutputStream =
        new DataOutputStream(new FileOutputStream(trainExamplesKeyPath))) {

      Nd4j.write(allBoardsKey, dataOutputStream);
    }

    try (DataOutputStream dataOutputStream =
        new DataOutputStream(new FileOutputStream(trainExamplesValuesPath))) {

      Nd4j.write(allValues, dataOutputStream);
    }
  }

  void resizeTrainExamplesHistory() {

    if (this.adversaryLearningConfiguration.getMaxTrainExamplesHistory() >=
        this.trainExamplesHistory.size()) {
      
      log.info("New train examples history map size {}",
          this.trainExamplesHistory.size());
      
      return;
    }

    int previousTrainExamplesSize = this.trainExamplesHistory.size();

    SortedSet<Integer> sortedIterationKeys = new TreeSet<>(this.trainExampleBoardsByIteration.keySet());
    Iterator<Integer> latestIterationIterator = sortedIterationKeys.iterator();
    
    StringBuilder removedIterations = new StringBuilder();
    while (this.trainExamplesHistory.size() > this.adversaryLearningConfiguration.getMaxTrainExamplesHistory()) {
      
      Integer remainingOldestIteration = latestIterationIterator.next();
      removedIterations.append(remainingOldestIteration).append(", ");
      Set<INDArray> boardExamplesToBeRemoves = this.trainExampleBoardsByIteration.get(remainingOldestIteration);
      
      boardExamplesToBeRemoves.stream().forEach(board -> this.trainExamplesHistory.remove(board));
      this.trainExampleBoardsByIteration.remove(remainingOldestIteration);
    }
    
    if (log.isInfoEnabled()) {
      log.info("Board examples from iteration[s] {} removed", removedIterations.substring(0, removedIterations.length() - 2));
      
      log.info(
          "Oldest from {} examples history removed to keep {} examples",
          previousTrainExamplesSize,
          this.trainExamplesHistory.size());
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

    return trainExamplesHistory;
  }

  public void setTrainExamplesHistory(Map<INDArray, AdversaryTrainingExample> trainExamplesHistory) {

    this.trainExamplesHistory = trainExamplesHistory;
  }

  public Map<Integer, Set<INDArray>> getTrainExampleBoardsByIteration() {

    return trainExampleBoardsByIteration;
  }

  public void setTrainExampleBoardsByIteration(Map<Integer, Set<INDArray>> trainExampleBoardsByIteration) {
 
    this.trainExampleBoardsByIteration = trainExampleBoardsByIteration;
  }
}
