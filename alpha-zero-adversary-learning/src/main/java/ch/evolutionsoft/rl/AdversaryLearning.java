package ch.evolutionsoft.rl;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
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
    loadEarlierTrainingExamples(adversaryLearningConfiguration.getTrainExamplesFileName());

    for (int iteration = adversaryLearningConfiguration.getIterationStart();
        iteration < adversaryLearningConfiguration.getIterationStart() + 
        adversaryLearningConfiguration.getNumberOfIterations();
        iteration++) {

      for (int episode = 1; episode <= adversaryLearningConfiguration.getNumberOfIterationsBeforePotentialUpdate(); episode++) {

        List<AdversaryTrainingExample> newExamples = this.executeEpisode(iteration);
  
        replaceOldTrainingExamplesWithNewActionProbabilities(newExamples);
  
        saveTrainExamplesHistory();
        
        log.info("Episode {}-{} ended, train examples {}", iteration, episode, this.trainExamplesHistory.size());
      }

      boolean updateAfterBetterPlayout = updateNeuralNet();

      if (adversaryLearningConfiguration.isAlwaysUpdateNeuralNetwork() || 
          updateAfterBetterPlayout) {

        log.info("Accepting new model");
        String absoluteBestModelPath =
            adversaryLearningConfiguration.getAbsoluteModelPathFrom(adversaryLearningConfiguration.getBestModelFileName());
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

  public List<AdversaryTrainingExample> executeEpisode(int iteration) {

    List<AdversaryTrainingExample> trainExamples = new ArrayList<>();

    Game currentGame = this.initialGame.createNewInstance();
    int currentPlayer = Game.MAX_PLAYER;

    this.mcts = new MonteCarloTreeSearch(computationGraph, adversaryLearningConfiguration);
    int moveNumber = 1;

    while (!currentGame.gameEnded()) {

      INDArray validMoves = currentGame.getValidMoves();
      Set<Integer> validMoveIndices = currentGame.getValidMoveIndices();

      INDArray actionProbabilities = this.mcts.getActionValues(currentGame,
            adversaryLearningConfiguration.getCurrentTemperature(iteration, moveNumber));
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

      updateMonteCarloSearchRoot(currentGame, moveAction);

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
          adversaryLearningConfiguration.getAbsoluteModelPathFrom(adversaryLearningConfiguration.getBestModelFileName());
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

  public Map<INDArray, AdversaryTrainingExample> loadEarlierTrainingExamples(String trainExamplesFile) throws IOException {

    if (restoreTrainingExamples) {
    
      try (ObjectInputStream trainExamplesInput = new ObjectInputStream(new FileInputStream(trainExamplesFile))) {
  
        Object readObject = trainExamplesInput.readObject();
        if (readObject instanceof List<?>) {
          
          List<AdversaryTrainingExample> storedExamples = (List<AdversaryTrainingExample>) readObject;
          
          for (AdversaryTrainingExample currentItem : storedExamples) {
            
            this.trainExamplesHistory.put(currentItem.getBoard(), currentItem);
          }
          
        } else if (readObject instanceof Map<?,?>) {
        
          this.trainExamplesHistory = (Map<INDArray, AdversaryTrainingExample>) readObject;
        }
  
        log.info("Restored train examples from {} with {} train examples",
            trainExamplesFile,
            this.trainExamplesHistory.size());
        
      } catch (ClassNotFoundException e) {
        log.warn(
            "Train examples from trainExamples.obj could not be restored. Continue with empty train examples history.",
            e);
      }
    }
    
    return this.trainExamplesHistory;
  }

  void replaceOldTrainingExamplesWithNewActionProbabilities(List<AdversaryTrainingExample> newExamples) {

    for (AdversaryTrainingExample currentExample : newExamples) {
      
      this.trainExamplesHistory.put(currentExample.getBoard(), currentExample);
    }
  }

  boolean updateNeuralNet() throws IOException {

    List<AdversaryTrainingExample> trainExamples = new ArrayList<>(this.trainExamplesHistory.values());
    Collections.shuffle(trainExamples);
    
    boolean updateAfterBetterPlayout = false;
    if (!adversaryLearningConfiguration.isAlwaysUpdateNeuralNetwork()) {

      String absoluteTempModelPath = adversaryLearningConfiguration.getAbsoluteModelPathFrom(TEMPMODEL_NAME);
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

      String bestModelPath = adversaryLearningConfiguration.getAbsoluteModelPathFrom(adversaryLearningConfiguration.getBestModelFileName());
      ModelSerializer.writeModel(computationGraph, bestModelPath.substring(0, bestModelPath.length() - ".bin".length()) + prependedZeros + iteration + ".bin", true);
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

  void updateMonteCarloSearchRoot(Game game, int moveAction) {

    try {
      this.mcts.updateWithMove(moveAction);

    } catch (IllegalArgumentException iae) {

      log.info("{}", game);
      throw new IllegalArgumentException(iae);
    }
  }

  void saveTrainExamplesHistory() throws IOException {

    this.resizeTrainExamplesHistory();

    String trainExamplesPath = adversaryLearningConfiguration.getAbsoluteModelPathFrom(
        adversaryLearningConfiguration.getTrainExamplesFileName());
    
    try (ObjectOutputStream trainExamplesOutput = new ObjectOutputStream(
        new FileOutputStream(trainExamplesPath))) {

      trainExamplesOutput.writeObject(trainExamplesHistory);

    }
  }

  void saveTrainExamplesHistory(int iteration) throws IOException {

    this.resizeTrainExamplesHistory();

    StringBuilder prependedZeros = prependZeros(iteration);
    String trainExamplesPath = adversaryLearningConfiguration.getAbsoluteModelPathFrom(
        adversaryLearningConfiguration.getTrainExamplesFileName());
    
    try (ObjectOutputStream trainExamplesOutput = new ObjectOutputStream(
        new FileOutputStream(trainExamplesPath.substring(0, trainExamplesPath.length() - ".obj".length())  + prependedZeros + iteration + ".obj"))) {

      trainExamplesOutput.writeObject(trainExamplesHistory);

    }
  }

  void resizeTrainExamplesHistory() {

    if (this.adversaryLearningConfiguration.getMaxTrainExamplesHistory() >=
        this.trainExamplesHistory.size()) {
      
      return;
    }

    Comparator<AdversaryTrainingExample> byIterationDescending = 
      (AdversaryTrainingExample firstExample, AdversaryTrainingExample secondExample) -> 
      secondExample.getIteration() - firstExample.getIteration();
    this.trainExamplesHistory =
        this.trainExamplesHistory.entrySet().stream().
        sorted(Entry.comparingByValue(byIterationDescending)).
        limit(this.adversaryLearningConfiguration.getMaxTrainExamplesHistory()).
        collect(Collectors.toMap(
           Entry::getKey, Entry::getValue, (e1, e2) -> e1, HashMap::new));
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

    log.info("Learning rate from computation graph model layer 'OutputLayer': {}",
        NetworkUtils.getLearningRate(computationGraph, "OutputLayer"));
    
    // The outputs from the fitted network will have new action probabilities
    this.mcts.resetStoredOutputs();

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
}
