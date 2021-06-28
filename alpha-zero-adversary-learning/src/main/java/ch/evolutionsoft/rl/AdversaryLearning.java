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
import java.util.List;
import java.util.Set;

import org.apache.commons.math3.distribution.EnumeratedIntegerDistribution;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import cc.mallet.types.Dirichlet;

public class AdversaryLearning {

  public static final double DRAW_VALUE = 0.5f;
  public static final double MAX_WIN = 1f;
  public static final double MIN_WIN = 0f;

  private static final Logger log = LoggerFactory.getLogger(AdversaryLearning.class);

  List<AdversaryTrainingExample> trainExamplesHistory = new ArrayList<>();
  
  Game game;
  
  ComputationGraph computationGraph;
  ComputationGraph pComputationGraph;
  
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
  
  List<AdversaryTrainingExample> executeEpisode(int iteration) {
    
    List<AdversaryTrainingExample> trainExamples = new ArrayList<>();
    
    INDArray currentBoard = game.getInitialBoard();
    int currentPlayer = Game.MAX_PLAYER;
    
    this.mcts = new MonteCarloSearch(game, computationGraph, adversaryLearningConfiguration, currentBoard);
    
    while (!game.gameEnded(currentBoard)) {

      INDArray validMoves = game.getValidMoves(currentBoard);
      Set<Integer> emptyFields = game.getValidMoveIndices(currentBoard);

      INDArray normalizedActionProbabilities = Nd4j.zeros(game.getFieldCount());
      if (hasMoreThanOneMove(emptyFields)) {

        INDArray actionProbabilities =
            this.mcts.getActionValues(
                currentBoard,
                adversaryLearningConfiguration.getCurrentTemperature(iteration, -1));
        INDArray validActionProbabilities = actionProbabilities.mul(validMoves);
        normalizedActionProbabilities = validActionProbabilities.div(Nd4j.sum(actionProbabilities));         
      }
      
      AdversaryTrainingExample trainingExample = new AdversaryTrainingExample(
          currentBoard,
          currentPlayer,
          normalizedActionProbabilities,
          iteration);

      trainExamples.remove(trainingExample);
      trainExamples.add(trainingExample);
      
      List<AdversaryTrainingExample> symmetries = game.getSymmetries(
          currentBoard.dup(),
          normalizedActionProbabilities.dup(),
          currentPlayer,
          iteration);

      Set<AdversaryTrainingExample> addedSymmetries = new HashSet<>();
      addedSymmetries.add(trainingExample);
      for (AdversaryTrainingExample symmetryExample : symmetries) {

        if (!addedSymmetries.contains(symmetryExample)) {
          trainExamples.remove(symmetryExample);
          trainExamples.add(symmetryExample);
          addedSymmetries.add(symmetryExample);
        }
      }

      int moveAction = -1;
      if (!hasMoreThanOneMove(emptyFields)) {
      
        moveAction = emptyFields.iterator().next();
      
      } else {
      
        double alpha = adversaryLearningConfiguration.getDirichletAlpha();
        Dirichlet dirichlet = new Dirichlet(emptyFields.size(), alpha);
        
        INDArray nextDistribution = Nd4j.createFromArray(dirichlet.nextDistribution());
        int[] validIndices = game.getValidIndices(emptyFields);
        INDArray reducedValidActionProbabilities = normalizedActionProbabilities.get(Nd4j.createFromArray(validIndices));
        INDArray noiseActionDistribution = reducedValidActionProbabilities.mul(1 - adversaryLearningConfiguration.getDirichletWeight()).add(
            nextDistribution.mul(adversaryLearningConfiguration.getDirichletWeight()));
        
        EnumeratedIntegerDistribution distribution =
            new EnumeratedIntegerDistribution(
                validIndices,
                noiseActionDistribution.toDoubleVector()
                );
        
        moveAction = distribution.sample();
        
        while (!emptyFields.contains(moveAction)) {
          // Not possible with reducedValidActionProbabilities above
          log.warn("Resample invalid random choice move.");
          moveAction = distribution.sample();
        }
      }
      
      currentBoard = game.makeMove(currentBoard, moveAction, currentPlayer);
      this.mcts.updateWithMove(moveAction);
      
      if (game.gameEnded(currentBoard)) {
        
        // Now the currentPlayer has moved, clarify with previousPlayer for clarifying gameResult
        int previousPlayer = currentPlayer;
        if (game.hasWon(currentBoard, previousPlayer)) {
  
          double gameResult = 0;
          
          for (AdversaryTrainingExample trainExample : trainExamples) {
            
            trainExample.setCurrentPlayerValue((float) (trainExample.getCurrentPlayer() == previousPlayer ? gameResult : 1 - gameResult));
          }
        } else {

          
          for (AdversaryTrainingExample trainExample : trainExamples) {
            
            trainExample.setCurrentPlayerValue((float) DRAW_VALUE);
          }
          
        }
        return trainExamples;
      }
      
      currentPlayer = currentPlayer == Game.MAX_PLAYER ?
          Game.MIN_PLAYER : Game.MAX_PLAYER;
    }
    
    return null;
  }
  
  public void performLearning() throws IOException {

    if (restoreTrainedNeuralNet) {
      
      this.pComputationGraph = ModelSerializer.restoreComputationGraph("tempmodel.bin", true);
      this.computationGraph = ModelSerializer.restoreComputationGraph("bestmodel.bin", true);
      log.info("restored bestmodel.bin and tempmodel.bin");
    }
 
    if (restoreTrainingExamples) {

      loadEarlierTrainingExamples();
    }
      
      for (int iteration = adversaryLearningConfiguration.getIterationStart();
          iteration < adversaryLearningConfiguration.getIterationStart() + adversaryLearningConfiguration.getNumberOfIterations();
          iteration++) {
        
        List<AdversaryTrainingExample> newExamples = this.executeEpisode(iteration);
          
        for (AdversaryTrainingExample trainExample : newExamples) {
  
          this.trainExamplesHistory.remove(trainExample);
          this.trainExamplesHistory.add(trainExample);
        }      
      
        while (this.trainExamplesHistory.size() > adversaryLearningConfiguration.getMaxTrainExamplesHistory()) {
          this.trainExamplesHistory.remove(0);
        }
        
        try (ObjectOutputStream trainExamplesOutput = new ObjectOutputStream(
            new FileOutputStream("trainExamples.obj"))) {
  
          trainExamplesOutput.writeObject(trainExamplesHistory);
        }
      
        List<AdversaryTrainingExample> trainExamples = new ArrayList<>(this.trainExamplesHistory);
        Collections.shuffle(trainExamples);
  
        boolean updateAfterBetterPlayout = false;
        if (!adversaryLearningConfiguration.isAlwaysUpdateNeuralNetwork() && iteration % adversaryLearningConfiguration.getNumberOfEpisodesBeforePotentialUpdate() == 0 ) {
        
          ModelSerializer.writeModel(computationGraph, "tempmodel.bin", true);
          this.pComputationGraph = ModelSerializer.restoreComputationGraph("tempmodel.bin", true);
          
          this.computationGraph = this.performTraining(this.computationGraph, trainExamples);
          
          AdversaryAgentDriver adversaryAgentDriver = new AdversaryAgentDriver(this.game, this.pComputationGraph, this.computationGraph);
          int[] gameResults = adversaryAgentDriver.playGames(adversaryLearningConfiguration, iteration);
          
          log.info("New model wins {} / prev model wins {} / draws {}", gameResults[1], gameResults[0], gameResults[2]);
          
          updateAfterBetterPlayout = 
              (gameResults[1] + 0.5 * gameResults[2]) /
              (double) (gameResults[0] + gameResults[1] + 0.5 * gameResults[2]) > adversaryLearningConfiguration.getUpdateGamesNewNetworkWinRatioThreshold();
              
           if (!updateAfterBetterPlayout) {
  
              log.info("Rejecting new model");
              this.computationGraph = ModelSerializer.restoreComputationGraph("tempmodel.bin", true);  
           } 
          
        } else if (iteration % adversaryLearningConfiguration.getNumberOfEpisodesBeforePotentialUpdate() == 0) {
  
          this.computationGraph = this.performTraining(this.computationGraph, trainExamples);
        }
  
        if ((adversaryLearningConfiguration.isAlwaysUpdateNeuralNetwork() && iteration % adversaryLearningConfiguration.getNumberOfEpisodesBeforePotentialUpdate() == 0 ) || updateAfterBetterPlayout) {
          
          log.info("Accepting new model");
          ModelSerializer.writeModel(computationGraph, "bestmodel.bin", true);
          if (updateAfterBetterPlayout) {
            game.evaluateOpeningAnswers(pComputationGraph);
          }
          game.evaluateOpeningAnswers(computationGraph);
          game.evaluateNetwork(computationGraph);
        
        }
        
        log.info("Iteration {} ended, train examples {}", iteration, this.trainExamplesHistory.size());
      
        if (0 == iteration % 1000) {
          
          ModelSerializer.writeModel(computationGraph, "2021-06-17-bestmodel0" + iteration + "a.bin", true);
        }
      
      }
    
  }

  boolean hasMoreThanOneMove(Set<Integer> emptyFields) {

    return 1 < emptyFields.size();
  }

  void loadEarlierTrainingExamples() throws IOException, FileNotFoundException {

    try (ObjectInputStream trainExamplesInput = new ObjectInputStream(
        new FileInputStream("trainExamples.obj"))) {
      
      this.trainExamplesHistory = (List<AdversaryTrainingExample>) trainExamplesInput.readObject();
      log.info("Restored train examples from trainEamples.obj with {} train examples",
          this.trainExamplesHistory.size());
      
    } catch (ClassNotFoundException e) {
      log.warn("Train examples from trainExamples.obj could not be restored. Continue with empty train examples history.", e);
    }
  }
  
  ComputationGraph performTraining(ComputationGraph computationGraph, List<AdversaryTrainingExample> trainingExamples) {

    INDArray inputBoards = Nd4j.zeros(trainingExamples.size(), 3, 3, 3);
    INDArray probabilitiesLabels = Nd4j.zeros(trainingExamples.size(), game.getFieldCount());
    INDArray valueLabels =  Nd4j.zeros(trainingExamples.size(), 1);    
    
    for (int exampleNumber = 0; exampleNumber < trainingExamples.size(); exampleNumber++) {
      
      AdversaryTrainingExample currentTrainingExample = trainingExamples.get(exampleNumber);
      inputBoards.putRow(exampleNumber, currentTrainingExample.getBoard());
      
      probabilitiesLabels.putRow(exampleNumber, currentTrainingExample.getActionIndexProbabilities());
      valueLabels.putRow(exampleNumber, Nd4j.create(1).putScalar(0, currentTrainingExample.getCurrentPlayerValue()));
    }

    MultiDataSet dataSet = new org.nd4j.linalg.dataset.MultiDataSet(new INDArray[] {inputBoards}, new INDArray[] {probabilitiesLabels, valueLabels});
    
    computationGraph.fit(dataSet);

    // The outputs from the fitted network will have new action probabilities
    this.mcts.resetStoredOutputs();
    
    return computationGraph;
  }
}
