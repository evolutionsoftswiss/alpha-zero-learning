package ch.evolutionsoft.rl;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;

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
import ch.evolutionsoft.net.game.NeuralNetConstants;

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
  public static final double MAX_WIN = NeuralNetConstants.ONE;
  public static final double MIN_WIN = NeuralNetConstants.ZERO;

  public static final int NO_MOVE = -2;
  
  public static final int SEVEN_DIGITS = 7;

  private static final Logger log = LoggerFactory.getLogger(AdversaryLearning.class);

  List<AdversaryTrainingExample> trainExamplesHistory = new ArrayList<>();

  Game game;

  ComputationGraph computationGraph;
  ComputationGraph previousComputationGraph;

  AdversaryLearningConfiguration adversaryLearningConfiguration;

  MonteCarloSearch mcts;

  boolean restoreTrainingExamples;

  boolean restoreTrainedNeuralNet;

  public AdversaryLearning(Game game, ComputationGraph computationGraph, AdversaryLearningConfiguration configuration) {

    this.game = game;
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

      List<AdversaryTrainingExample> newExamples = this.executeEpisode(iteration);

      replaceOldTrainingExamplesWithNewActionProbabilities(newExamples);

      saveTrainExamplesHistory(-1);

      log.info("Iteration {} ended, train examples {}", iteration, this.trainExamplesHistory.size());

      boolean updateAfterBetterPlayout = updateNeuralNet(iteration);

      if ((adversaryLearningConfiguration.isAlwaysUpdateNeuralNetwork() && 
          iteration % adversaryLearningConfiguration.getNumberOfIterationsBeforePotentialUpdate() == 0) || 
          updateAfterBetterPlayout) {

        log.info("Accepting new model");
        ModelSerializer.writeModel(computationGraph, "bestmodel.bin", true);
        if (updateAfterBetterPlayout) {
          game.evaluateBoardActionExamples(previousComputationGraph);
        }
        game.evaluateBoardActionExamples(computationGraph);
        game.evaluateNetwork(computationGraph);

      }

      createCheckpoint(iteration);
    }
  }

  List<AdversaryTrainingExample> executeEpisode(int iteration) {

    List<AdversaryTrainingExample> trainExamples = new ArrayList<>();

    Object savedPosition = game.savePosition();
    INDArray currentBoard = game.getInitialBoard();
    int currentPlayer = Game.MAX_PLAYER;

    this.mcts = new MonteCarloSearch(game, computationGraph, adversaryLearningConfiguration);

    while (!game.gameEnded(currentBoard)) {

      INDArray validMoves = game.getValidMoves(currentBoard);
      Set<Integer> validMoveIndices = game.getValidMoveIndices(currentBoard);

      INDArray actionProbabilities = this.mcts.getActionValues(currentBoard,
            adversaryLearningConfiguration.getCurrentTemperature(iteration, NO_MOVE));
      INDArray validActionProbabilities = actionProbabilities.mul(validMoves);
      INDArray normalizedActionProbabilities = validActionProbabilities.div(Nd4j.sum(actionProbabilities));

      List<AdversaryTrainingExample> newTrainingExamples = 
          createNewTrainingExamplesWithSymmetries(iteration, currentBoard, currentPlayer,
              normalizedActionProbabilities);

      trainExamples.removeAll(newTrainingExamples);
      trainExamples.addAll(newTrainingExamples);
      
      int moveAction = chooseNewMoveAction(validMoveIndices, normalizedActionProbabilities);

      currentBoard = game.makeMove(currentBoard, moveAction, currentPlayer);

      updateMonteCarloSearchRoot(currentBoard, moveAction);

      handleGameEnded(trainExamples, currentBoard, currentPlayer);

      currentPlayer = currentPlayer == Game.MAX_PLAYER ? Game.MIN_PLAYER : Game.MAX_PLAYER;
    }

    game.restorePosition(savedPosition);

    return trainExamples;
  }

  private void loadComputationGraphs() throws IOException {

    if (restoreTrainedNeuralNet) {

      this.computationGraph = ModelSerializer.restoreComputationGraph("bestmodel.bin", false);
      this.computationGraph.setLearningRate(this.adversaryLearningConfiguration.getLearningRate());
      log.info("restored bestmodel.bin");

      if (!this.adversaryLearningConfiguration.isAlwaysUpdateNeuralNetwork()) {

        this.previousComputationGraph = ModelSerializer.restoreComputationGraph("tempmodel.bin", false);
        this.previousComputationGraph.setLearningRate(this.adversaryLearningConfiguration.getLearningRate());
        log.info("restored tempmodel.bin");
      }
    }
  }

  private void loadEarlierTrainingExamples() throws IOException, FileNotFoundException {

    if (restoreTrainingExamples) {
    
      try (ObjectInputStream trainExamplesInput = new ObjectInputStream(new FileInputStream("trainExamples.obj"))) {
  
        this.trainExamplesHistory = (List<AdversaryTrainingExample>) trainExamplesInput.readObject();
        log.info("Restored train examples from trainExamples.obj with {} train examples",
            this.trainExamplesHistory.size());
  
      } catch (ClassNotFoundException e) {
        log.warn(
            "Train examples from trainExamples.obj could not be restored. Continue with empty train examples history.",
            e);
      }
    }
  }

  private void replaceOldTrainingExamplesWithNewActionProbabilities(List<AdversaryTrainingExample> newExamples) {

    this.trainExamplesHistory.removeAll(newExamples);
    this.trainExamplesHistory.addAll(newExamples);
  }

  private boolean updateNeuralNet(int iteration) throws IOException {

    List<AdversaryTrainingExample> trainExamples = new ArrayList<>(this.trainExamplesHistory);
    Collections.shuffle(trainExamples);
    
    boolean updateAfterBetterPlayout = false;
    if (!adversaryLearningConfiguration.isAlwaysUpdateNeuralNetwork()
        && iteration % adversaryLearningConfiguration.getNumberOfIterationsBeforePotentialUpdate() == 0) {

      ModelSerializer.writeModel(computationGraph, "tempmodel.bin", true);
      this.previousComputationGraph = ModelSerializer.restoreComputationGraph("tempmodel.bin", true);

      this.computationGraph = this.fitNeuralNet(this.computationGraph, trainExamples);

      AdversaryAgentDriver adversaryAgentDriver = new AdversaryAgentDriver(this.game, this.previousComputationGraph,
          this.computationGraph);

      int[] gameResults = adversaryAgentDriver.playGames(adversaryLearningConfiguration, iteration);

      log.info("New model wins {} / prev model wins {} / draws {}", gameResults[1], gameResults[0], gameResults[2]);

      updateAfterBetterPlayout = (gameResults[1] + 0.5 * gameResults[2])
          / (double) (gameResults[0] + gameResults[1] + 0.5 * gameResults[2]) > adversaryLearningConfiguration
              .getUpdateGamesNewNetworkWinRatioThreshold();

      if (!updateAfterBetterPlayout) {

        log.info("Rejecting new model");
        this.computationGraph = ModelSerializer.restoreComputationGraph("tempmodel.bin", true);
      }

    } else if (iteration % adversaryLearningConfiguration.getNumberOfIterationsBeforePotentialUpdate() == 0) {

      this.computationGraph = this.fitNeuralNet(this.computationGraph, trainExamples);
    }

    return updateAfterBetterPlayout;
  }

  private void createCheckpoint(int iteration) throws IOException, FileNotFoundException {

    int prependingZeros = SEVEN_DIGITS - String.valueOf(iteration).length();
    
    String prependedZeros = "";
    for (int n = 1; n <= prependingZeros; n++) {
      prependedZeros += "0";
    }
    
    if (0 == iteration % adversaryLearningConfiguration.getCheckPointIterationsFrequency()) {

      ModelSerializer.writeModel(computationGraph, "bestmodel" + prependedZeros + iteration + ".bin", true);
      saveTrainExamplesHistory(iteration);
    }
  }

  private List<AdversaryTrainingExample> createNewTrainingExamplesWithSymmetries(int iteration,
      INDArray currentBoard, int currentPlayer, INDArray normalizedActionProbabilities) {

    List<AdversaryTrainingExample> newTrainingExamples = new ArrayList<>();
    
    AdversaryTrainingExample trainingExample = new AdversaryTrainingExample(currentBoard, currentPlayer,
        normalizedActionProbabilities, iteration);

    newTrainingExamples.add(trainingExample);

    List<AdversaryTrainingExample> symmetries = game.getSymmetries(currentBoard.dup(),
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

  private int chooseNewMoveAction(Set<Integer> validMoveIndices, INDArray normalizedActionProbabilities) {

    int moveAction = NO_MOVE;
    if (!hasMoreThanOneMove(validMoveIndices)) {

      moveAction = validMoveIndices.iterator().next();

    } else {

      double alpha = adversaryLearningConfiguration.getDirichletAlpha();
      Dirichlet dirichlet = new Dirichlet(validMoveIndices.size(), alpha);

      INDArray nextDistribution = Nd4j.createFromArray(dirichlet.nextDistribution());
      int[] validIndices = game.getValidIndices(validMoveIndices);
      INDArray reducedValidActionProbabilities = normalizedActionProbabilities.get(Nd4j.createFromArray(validIndices));
      INDArray noiseActionDistribution = reducedValidActionProbabilities
          .mul(1 - adversaryLearningConfiguration.getDirichletWeight())
          .add(nextDistribution.mul(adversaryLearningConfiguration.getDirichletWeight()));

      EnumeratedIntegerDistribution distribution = new EnumeratedIntegerDistribution(validIndices,
          noiseActionDistribution.toDoubleVector());

      moveAction = distribution.sample();

      while (!validMoveIndices.contains(moveAction)) {
        // Not possible with correctly reducedValidActionProbabilities above
        log.warn("Resample invalid random choice move: {} \nvalidIndices {}\nreducedActionProbs{}", moveAction,
            validIndices, reducedValidActionProbabilities);
        moveAction = distribution.sample();
      }
    }
    return moveAction;
  }

  private void handleGameEnded(List<AdversaryTrainingExample> trainExamples, INDArray currentBoard, int currentPlayer) {

    if (game.gameEnded(currentBoard)) {

      // Now the currentPlayer has moved, clarify with previousPlayer for clarifying
      // gameResult
      int previousPlayer = currentPlayer;
      if (game.hasWon(currentBoard, previousPlayer)) {

        double gameResult = 0;

        for (AdversaryTrainingExample trainExample : trainExamples) {

          trainExample.setCurrentPlayerValue(
              (float) (trainExample.getCurrentPlayer() == previousPlayer ? gameResult : 1 - gameResult));
        }
      } else {

        for (AdversaryTrainingExample trainExample : trainExamples) {

          trainExample.setCurrentPlayerValue((float) DRAW_VALUE);
        }
      }
    }
  }

  private void updateMonteCarloSearchRoot(INDArray currentBoard, int moveAction) {

    try {
      this.mcts.updateWithMove(moveAction);

    } catch (IllegalArgumentException iae) {

      log.info("{}", game);
      log.info("{}", currentBoard);
      throw new RuntimeException(iae);
    }
  }

  private void saveTrainExamplesHistory(int iteration) throws IOException, FileNotFoundException {

    if (this.trainExamplesHistory.size() > adversaryLearningConfiguration.getMaxTrainExamplesHistory()) {

      this.trainExamplesHistory
          .subList(0, trainExamplesHistory.size() - adversaryLearningConfiguration.getMaxTrainExamplesHistory())
          .clear();
    }

    try (ObjectOutputStream trainExamplesOutput = new ObjectOutputStream(
        new FileOutputStream("trainExamples" + (iteration > 0 ? "000" + iteration : "") + ".obj"))) {

      trainExamplesOutput.writeObject(trainExamplesHistory);
    }
  }

  boolean hasMoreThanOneMove(Set<Integer> emptyFields) {

    return 1 < emptyFields.size();
  }

  ComputationGraph fitNeuralNet(ComputationGraph computationGraph, List<AdversaryTrainingExample> trainingExamples) {

    int batchSize = adversaryLearningConfiguration.getBatchSize();
    int trainingExamplesSize = trainingExamples.size();
    int batchNumber = 1 + trainingExamplesSize / batchSize;
    
    long[] gameInputBoardStackShape = game.getInitialBoard().shape();
    
    List<MultiDataSet> batchedMultiDataSet = new LinkedList<MultiDataSet>();

    for (int currentBatch = 0; currentBatch < batchNumber; currentBatch++) {

      INDArray inputBoards = Nd4j.zeros(batchSize, gameInputBoardStackShape[0], gameInputBoardStackShape[1],
          gameInputBoardStackShape[2]);
      INDArray probabilitiesLabels = Nd4j.zeros(batchSize, game.getNumberOfAllAvailableMoves());
      INDArray valueLabels = Nd4j.zeros(batchSize, 1);
      
      if (currentBatch >= batchNumber - 1) {

        int lastBatchSize = trainingExamplesSize % batchSize;
        inputBoards = Nd4j.zeros(lastBatchSize, gameInputBoardStackShape[0], gameInputBoardStackShape[1],
        gameInputBoardStackShape[2]);
        probabilitiesLabels = Nd4j.zeros(lastBatchSize, game.getNumberOfAllAvailableMoves());
        valueLabels = Nd4j.zeros(lastBatchSize, 1);
      }

      for (int batchExample = 0, exampleNumber = currentBatch * batchSize;
          exampleNumber < (currentBatch + 1) * batchSize && exampleNumber < trainingExamplesSize;
          exampleNumber++, batchExample++) {
        
        AdversaryTrainingExample currentTrainingExample = trainingExamples.get(exampleNumber);
        inputBoards.putRow(batchExample, currentTrainingExample.getBoard());
  
        INDArray actionIndexProbabilities = Nd4j.zeros(game.getNumberOfAllAvailableMoves());
        INDArray trainingExampleActionProbabilities = currentTrainingExample.getActionIndexProbabilities();
        if (actionIndexProbabilities.shape()[0] > trainingExampleActionProbabilities.shape()[0]) {
  
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
  
          actionIndexProbabilities = trainingExampleActionProbabilities;
        }
  
        probabilitiesLabels.putRow(batchExample, actionIndexProbabilities);
  
        valueLabels.putRow(batchExample, Nd4j.create(1).putScalar(0, currentTrainingExample.getCurrentPlayerValue()));
      }
      
      batchedMultiDataSet.add( new org.nd4j.linalg.dataset.MultiDataSet(new INDArray[] { inputBoards },
      new INDArray[] { probabilitiesLabels, valueLabels }));
    }

    for (int batchIteration = 0; batchIteration < batchNumber; batchIteration++) {
      computationGraph.fit(batchedMultiDataSet.get(batchIteration));
      log.info("Batch size from computation graph model {}", computationGraph.batchSize());
    }

    log.info("Learning rate from computation graph model layer 'OutputLayer': {}",
        NetworkUtils.getLearningRate(computationGraph, "OutputLayer"));
    
    // The outputs from the fitted network will have new action probabilities
    this.mcts.resetStoredOutputs();

    return computationGraph;
  }
}
