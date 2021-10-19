package ch.evolutionsoft.rl;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
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
import org.deeplearning4j.util.NetworkUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
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

  public static final Logger log = LoggerFactory.getLogger(AdversaryLearning.class);
  
  Map<INDArray, AdversaryTrainingExample> trainExamplesHistory = new HashMap<>();

  Map<Integer, Set<INDArray>> trainExampleBoardsByIteration = new HashMap<>();

  Game initialGame;

  ComputationGraph computationGraph;
  ComputationGraph previousComputationGraph;

  AdversaryLearningConfiguration adversaryLearningConfiguration;

  MonteCarloTreeSearch mcts;

  boolean restoreTrainingExamples;

  boolean restoreTrainedNeuralNet;

  public AdversaryLearning(Game game, ComputationGraph computationGraph, AdversaryLearningConfiguration configuration) {

    this.initialGame = game;
    this.computationGraph = computationGraph;
    this.adversaryLearningConfiguration = configuration;
    this.restoreTrainingExamples = configuration.getIterationStart() > 1;
    this.restoreTrainedNeuralNet = configuration.getIterationStart() > 1;
    log.info("Using configuration\n{}", configuration);
  }

  public void performLearning() throws IOException {

    loadComputationGraphs();
    loadEarlierTrainingExamples();

    for (int iteration = adversaryLearningConfiguration.getIterationStart();
        iteration < adversaryLearningConfiguration.getIterationStart() + 
        adversaryLearningConfiguration.getNumberOfIterations();
        iteration++) {
      
      this.trainExampleBoardsByIteration.put(iteration, new HashSet<>());
      final int currentIteration = iteration;
      ThreadPoolExecutor executor = (ThreadPoolExecutor) Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors() / 2);
      CompletionService<List<AdversaryTrainingExample>> completionService = new ExecutorCompletionService<>(executor);
      
      Set<AdversaryTrainingExample> examplesFromEpisodes = new HashSet<>();
      
      for (int episode = 1; episode <= adversaryLearningConfiguration.getNumberOfIterationsBeforePotentialUpdate(); episode++) {

        final MonteCarloTreeSearch mcts = new MonteCarloTreeSearch(
            computationGraph.clone(),
            null);
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
        while(received < adversaryLearningConfiguration.getNumberOfIterationsBeforePotentialUpdate()) {
 
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
      
      log.info("Iteration {} ended", iteration);
    }
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

      double currentTemperature = 1.0;
      if (moveNumber >= 10) {
        
        currentTemperature = 0.0;
      }
      
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

  public void loadComputationGraphs() throws IOException {

    if (restoreTrainedNeuralNet) {

      String absoluteBestModelPath =
          adversaryLearningConfiguration.getAbsolutePathFrom(adversaryLearningConfiguration.getBestModelFileName());
      this.computationGraph = ModelSerializer.restoreComputationGraph(absoluteBestModelPath, true);
      this.computationGraph.setLearningRate(this.adversaryLearningConfiguration.getLearningRate());
      if (null != this.adversaryLearningConfiguration.getLearningRateSchedule()) {
        this.computationGraph.setLearningRate(this.adversaryLearningConfiguration.getLearningRateSchedule());
      }
      log.info("restored model {}", absoluteBestModelPath);

      if (!this.adversaryLearningConfiguration.isAlwaysUpdateNeuralNetwork()) {

        this.previousComputationGraph = ModelSerializer.restoreComputationGraph(absoluteBestModelPath, true);
        this.previousComputationGraph.setLearningRate(this.adversaryLearningConfiguration.getLearningRate());
        if (null != this.adversaryLearningConfiguration.getLearningRateSchedule()) {
          this.computationGraph.setLearningRate(this.adversaryLearningConfiguration.getLearningRateSchedule());
        }
        log.info("restored temp model from {}", absoluteBestModelPath);
      }
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
      log.info("exampleBoardsByIteration map has {} restored Set of boards entries with totall {} examples", this.trainExampleBoardsByIteration.size(), this.countAllExampleBoardsByIteration());
    }
  }

  void loadMapFromFile(String trainExamplesFile, Map<?, ?> targetMap) throws IOException {

    try (ObjectInputStream trainExamplesInput =
        new ObjectInputStream(
            new FileInputStream(trainExamplesFile))) {
 
      Object readObject = trainExamplesInput.readObject();
      if (readObject instanceof Map<?, ?>) {
      
        targetMap.putAll((Map) readObject);
      
      }
      
      int size = targetMap.size();
      log.info("Restored train examples from {} with {} train examples",
          trainExamplesFile,
          size);
      
    } catch (ClassNotFoundException e) {
      log.warn(
          String.format("Train examples from {} could not be restored. Continue with empty train examples history." ,
              this.adversaryLearningConfiguration.getAbsolutePathFrom(trainExamplesFile)),
          e);
    }
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
      log.debug("Updated {} examples with same board from earlier iterations, remaining {} examples may still duplicate known examples from running iteration",
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

    List<AdversaryTrainingExample> trainExamples = new ArrayList<>(this.trainExamplesHistory.values());
    Collections.shuffle(trainExamples);
    
    boolean updateAfterBetterPlayout = false;
    if (!adversaryLearningConfiguration.isAlwaysUpdateNeuralNetwork()) {

      String absoluteTempModelPath = adversaryLearningConfiguration.getAbsolutePathFrom(TEMPMODEL_NAME);
      ModelSerializer.writeModel(computationGraph, absoluteTempModelPath, true);
      
      log.info("Write temp model {}", absoluteTempModelPath);
      
      this.previousComputationGraph = ModelSerializer.restoreComputationGraph(absoluteTempModelPath, true);

      this.computationGraph = this.fitNeuralNet(this.computationGraph, trainExamples);

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

      this.computationGraph = this.fitNeuralNet(this.computationGraph, trainExamples);
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
        log.error("no child with move " + moveAction +
                "found for current root node with last move" + lastRoot.lastMove);
        
        return null;
      }
  }

  void saveTrainExamplesHistory() throws IOException {

    this.resizeTrainExamplesHistory();

    String trainExamplesByBoardPath = adversaryLearningConfiguration.getAbsolutePathFrom(
        adversaryLearningConfiguration.getTrainExamplesFileName());
    writeMapToFile(trainExamplesByBoardPath, this.trainExamplesHistory);
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
    
    String trainExamplesCheckpointByBoardFile = trainExamplesBasePath + 
        prependedZeros + iteration + suffix;
    this.writeMapToFile(trainExamplesCheckpointByBoardFile, this.trainExamplesHistory);
  }

  void writeMapToFile(String trainExamplesPath, Map<?, ?> sourceMap) throws IOException {

    try (ObjectOutputStream trainExamplesOutput = new ObjectOutputStream(
        new FileOutputStream(trainExamplesPath))) {

      trainExamplesOutput.writeObject(sourceMap);

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
    
    String removedIterations = "";
    while (this.trainExamplesHistory.size() > this.adversaryLearningConfiguration.getMaxTrainExamplesHistory()) {
      
      Integer remainingOldestIteration = latestIterationIterator.next();
      removedIterations += remainingOldestIteration + ", ";
      Set<INDArray> boardExamplesToBeRemoves = this.trainExampleBoardsByIteration.get(remainingOldestIteration);
      
      boardExamplesToBeRemoves.stream().forEach(board -> this.trainExamplesHistory.remove(board));
      this.trainExampleBoardsByIteration.remove(remainingOldestIteration);
      
      log.info("Board examples from iteration[s] {} removed", removedIterations.substring(0, removedIterations.length() - 2));
    }
    
    log.info(
        "Oldest from {} examples history removed to keep {} examples",
        previousTrainExamplesSize,
        this.trainExamplesHistory.size());
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

  ComputationGraph fitNeuralNet(ComputationGraph computationGraph, List<AdversaryTrainingExample> trainingExamples) {

    int batchSize = adversaryLearningConfiguration.getBatchSize();
    int trainingExamplesSize = trainingExamples.size();
    int batchNumber = 1 + trainingExamplesSize / batchSize;
    
    List<MultiDataSet> batchedMultiDataSet = createMiniBatchList(trainingExamples);

    for (int batchIteration = 0; batchIteration < batchNumber; batchIteration++) {

      computationGraph.fit(batchedMultiDataSet.get(batchIteration));
      
      if (0 == batchIteration && batchNumber > batchIteration + 1) {

        log.info("Batch size for {} batches from computation graph model {}", 
            batchNumber - 1,
            computationGraph.batchSize());
        
      } else if (batchNumber == batchIteration + 1) {

        log.info("{}. batch size from computation graph model {}",
            batchIteration + 1,
            computationGraph.batchSize());
      }
    }

    log.info("Iterations (number of updates) from computation graph model is {}",
        computationGraph.getIterationCount());
    log.info("Learning rate from computation graph model layer 'OutputLayer': {}",
        NetworkUtils.getLearningRate(computationGraph, "OutputLayer"));
    
    // The outputs from the fitted network will have new action probabilities
    // TODO clean up this.mcts.resetStoredOutputs();

    return computationGraph;
  }

  List<MultiDataSet> createMiniBatchList(List<AdversaryTrainingExample> trainingExamples) {
 
    int batchSize = adversaryLearningConfiguration.getBatchSize();
    int trainingExamplesSize = trainingExamples.size();
    int batchNumber = 1 + trainingExamplesSize / batchSize;
    if (0 == trainingExamplesSize % batchSize) {
      batchNumber--;
    }
 
    long[] gameInputBoardStackShape = initialGame.getInitialBoard().shape();
    
    List<MultiDataSet> batchedMultiDataSet = new LinkedList<>();

    for (int currentBatch = 0; currentBatch < batchNumber; currentBatch++) {

      INDArray inputBoards = Nd4j.zeros(batchSize, gameInputBoardStackShape[0], gameInputBoardStackShape[1],
          gameInputBoardStackShape[2]);
      INDArray probabilitiesLabels = Nd4j.zeros(batchSize, initialGame.getNumberOfAllAvailableMoves());
      INDArray valueLabels = Nd4j.zeros(batchSize, 1);
      
      if (currentBatch >= batchNumber - 1) {

        int lastBatchSize = trainingExamplesSize % batchSize;
        inputBoards = Nd4j.zeros(lastBatchSize, gameInputBoardStackShape[0], gameInputBoardStackShape[1],
        gameInputBoardStackShape[2]);
        probabilitiesLabels = Nd4j.zeros(lastBatchSize, initialGame.getNumberOfAllAvailableMoves());
        valueLabels = Nd4j.zeros(lastBatchSize, 1);
      }

      for (int batchExample = 0, exampleNumber = currentBatch * batchSize;
          exampleNumber < (currentBatch + 1) * batchSize && exampleNumber < trainingExamplesSize;
          exampleNumber++, batchExample++) {
        
        AdversaryTrainingExample currentTrainingExample = trainingExamples.get(exampleNumber);
        inputBoards.putRow(batchExample, currentTrainingExample.getBoard());
  
        INDArray actionIndexProbabilities = Nd4j.zeros(initialGame.getNumberOfAllAvailableMoves());
        INDArray trainingExampleActionProbabilities = currentTrainingExample.getActionIndexProbabilities();

        // TODO review simplification by always having getNumberOfAllAvailableMoves
        if (actionIndexProbabilities.shape()[0] > trainingExampleActionProbabilities.shape()[0]) {
  
          // Leave remaining moves at the end with 0, only pass at numberOfSquares in Go
          for (int i = 0; i < trainingExampleActionProbabilities.shape()[0]; i++) {
            actionIndexProbabilities.putScalar(i, trainingExampleActionProbabilities.getDouble(i));
          }
  
        } else if (actionIndexProbabilities.shape()[0] < currentTrainingExample.getActionIndexProbabilities()
            .shape()[0]) {
  
          throw new IllegalArgumentException(
              "Training example has more action than maximally specified by game.getNumberOfAllAvailableMoves()\n"
                  + "Max specified shape is " + actionIndexProbabilities.shape()[0] + " versus training example "
                  + currentTrainingExample.getActionIndexProbabilities());
  
        } else {
  
          // Shapes do match
          actionIndexProbabilities = trainingExampleActionProbabilities;
        }
  
        probabilitiesLabels.putRow(batchExample, actionIndexProbabilities);
  
        valueLabels.putRow(batchExample, Nd4j.zeros(1).putScalar(0, currentTrainingExample.getCurrentPlayerValue()));
      }
      
      batchedMultiDataSet.add( new org.nd4j.linalg.dataset.MultiDataSet(new INDArray[] { inputBoards },
      new INDArray[] { probabilitiesLabels, valueLabels }));
    }

    return batchedMultiDataSet;
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
