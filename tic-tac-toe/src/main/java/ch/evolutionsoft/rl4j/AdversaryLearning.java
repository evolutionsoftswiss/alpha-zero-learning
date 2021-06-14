package ch.evolutionsoft.rl4j;

import static ch.evolutionsoft.net.game.tictactoe.TicTacToeConstants.IMAGE_CHANNELS;
import static ch.evolutionsoft.net.game.tictactoe.TicTacToeConstants.IMAGE_SIZE;
import static ch.evolutionsoft.net.game.tictactoe.TicTacToeConstants.OCCUPIED_IMAGE_POINT;

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
import ch.evolutionsoft.net.game.tictactoe.TicTacToeConstants;
import ch.evolutionsoft.rl4j.tictactoe.EvaluationMain;
import ch.evolutionsoft.rl4j.tictactoe.TicTacToe;

public class AdversaryLearning {

  public static final double DRAW_VALUE = 0.5f;
  public static final double MAX_WIN = 1f;
  public static final double MIN_WIN = 0f;

  private static final Logger log = LoggerFactory.getLogger(AdversaryLearning.class);

  List<AdversaryTrainingExample> trainExamplesHistory = new ArrayList<>();
  
  ComputationGraph computationGraph;
  ComputationGraph pComputationGraph;
  
  int numberOfEpisodesUpdate;
  
  int maxTrainExamplesHistory = 5000;
  
  MonteCarloSearch mcts;
  
  int iterationStart = 1;
  
  int iterationSteps = 4000;
  
  double temperature = 1;

  boolean restoreTrainingExamples = iterationStart > 1;

  boolean restoreTrainedNeuralNet = iterationStart > 1;
  
  boolean alwaysUpdateNeuralNet = false;

  public AdversaryLearning(ComputationGraph computationGraph, int numberOfEpisodes) {
    
    this.computationGraph = computationGraph;
    this.numberOfEpisodesUpdate = numberOfEpisodes;
  }
  
  List<AdversaryTrainingExample> executeEpisode(int iteration) {
    
    List<AdversaryTrainingExample> trainExamples = new ArrayList<>();
    
    INDArray currentBoard = TicTacToeConstants.EMPTY_CONVOLUTIONAL_PLAYGROUND;
    int currentPlayer = TicTacToeConstants.MAX_PLAYER_CHANNEL;
    
    this.mcts = new MonteCarloSearch(computationGraph, currentBoard);
    
    while (!TicTacToe.gameEnded(currentBoard)) {

      INDArray validMoves = TicTacToe.getValidMoves(currentBoard);
      Set<Integer> emptyFields = TicTacToe.getEmptyFields(currentBoard);

      INDArray actionProbabilities = this.mcts.getActionValues(currentBoard, temperature);
      INDArray validActionProbabilities = actionProbabilities.mul(validMoves);
      INDArray normalizedActionProbabilities = validActionProbabilities.div(Nd4j.sum(actionProbabilities));
      
      AdversaryTrainingExample trainingExample = new AdversaryTrainingExample(
          currentBoard,
          currentPlayer,
          normalizedActionProbabilities,
          iteration);

      trainExamples.remove(trainingExample);
      trainExamples.add(trainingExample);
      
      List<AdversaryTrainingExample> symmetries = TicTacToe.getSymmetries(
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

      double alpha = 0.25;
      Dirichlet dirichlet = new Dirichlet(
          Nd4j.ones(TicTacToeConstants.COLUMN_COUNT).mul(validMoves).add(1e-10).
          mul(alpha).toDoubleVector());
      
      INDArray nextDistribution = Nd4j.createFromArray(dirichlet.nextDistribution()); 
      INDArray noiseActionDistribution = normalizedActionProbabilities.mul(0.75).add(
          nextDistribution.mul(0.25));
      
      noiseActionDistribution.div(noiseActionDistribution.sum(0));
      
      EnumeratedIntegerDistribution d =
          new EnumeratedIntegerDistribution(
              TicTacToe.COLUMN_INDICES,
              noiseActionDistribution.toDoubleVector()
              );
      
      int moveAction = d.sample();
 
      while (!emptyFields.contains(moveAction)) {
        log.warn("Resample invalid random choice move.");
        moveAction = d.sample();
      }
      
      currentBoard = TicTacToe.makeMove(currentBoard, moveAction, currentPlayer);
      this.mcts.updateWithMove(moveAction);
      
      if (TicTacToe.gameEnded(currentBoard)) {
        
        // Now the currentPlayer has moved, clarify with previousPlayer for clarifying gameResult
        int previousPlayer = currentPlayer;
        if (TicTacToe.hasWon(currentBoard, previousPlayer)) {
  
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
      
      currentPlayer = currentPlayer == TicTacToeConstants.MAX_PLAYER_CHANNEL ?
          TicTacToeConstants.MIN_PLAYER_CHANNEL : TicTacToeConstants.MAX_PLAYER_CHANNEL;
    }
    
    return null;
  }
  
  public void performLearning() throws IOException {

    if (restoreTrainedNeuralNet) {
      
      this.pComputationGraph = ModelSerializer.restoreComputationGraph("tempmodel.bin", true);
      this.computationGraph = ModelSerializer.restoreComputationGraph("bestmodel.bin", true);
      
    }
 
    if (restoreTrainingExamples) {

      loadEarlierTrainingExamples();
    }
      
      for (int iteration = iterationStart; iteration < iterationStart + iterationSteps; iteration++) {
          
        List<AdversaryTrainingExample> newExamples = this.executeEpisode(iteration);
          
        for (AdversaryTrainingExample trainExample : newExamples) {
  
          this.trainExamplesHistory.remove(trainExample);
          this.trainExamplesHistory.add(trainExample);
        }      
      
        while (this.trainExamplesHistory.size() > this.maxTrainExamplesHistory) {
          this.trainExamplesHistory.remove(0);
        }
        
        try (ObjectOutputStream trainExamplesOutput = new ObjectOutputStream(
            new FileOutputStream("trainExamples.obj"))) {
  
          trainExamplesOutput.writeObject(trainExamplesHistory);
        }
      
        List<AdversaryTrainingExample> trainExamples = new ArrayList<>(this.trainExamplesHistory);
        Collections.shuffle(trainExamples);
  
        boolean updateAfterBetterPlayout = false;
        if (!alwaysUpdateNeuralNet && iteration % this.numberOfEpisodesUpdate == 0 ) {
        
          ModelSerializer.writeModel(computationGraph, "tempmodel.bin", true);
          this.pComputationGraph = ModelSerializer.restoreComputationGraph("tempmodel.bin", true);
          
          this.computationGraph = this.performTraining(this.computationGraph, trainExamples);
          
          AdversaryAgentDriver adversaryAgentDriver = new AdversaryAgentDriver(this.pComputationGraph, this.computationGraph);
          int[] gameResults = adversaryAgentDriver.playGames(36, TicTacToeConstants.EMPTY_CONVOLUTIONAL_PLAYGROUND, temperature);
          
          log.info("New model wins {} / prev model wins {} / draws {}", gameResults[1], gameResults[0], gameResults[2]);
          
          updateAfterBetterPlayout = 
              (gameResults[1] + 0.5 * gameResults[2]) /
              (double) (gameResults[0] + gameResults[1] + 0.5 * gameResults[2]) > 0.55;
              
           if (!updateAfterBetterPlayout) {
  
              log.info("Rejecting new model");
              this.computationGraph = ModelSerializer.restoreComputationGraph("tempmodel.bin", true);  
           } 
          
        } else if (iteration % this.numberOfEpisodesUpdate == 0) {
  
          this.computationGraph = this.performTraining(this.computationGraph, trainExamples);
        }
  
        if ((alwaysUpdateNeuralNet && iteration % this.numberOfEpisodesUpdate == 0 ) || updateAfterBetterPlayout) {
          
          log.info("Accepting new model");
          ModelSerializer.writeModel(computationGraph, "bestmodel.bin", true);
          if (updateAfterBetterPlayout) {
            evaluateOpeningAnswers(pComputationGraph);
          }
          evaluateOpeningAnswers(computationGraph);
          EvaluationMain.evaluateNetwork(computationGraph);
        
        }
        
        log.info("Iteration {} ended, train examples {}", iteration, this.trainExamplesHistory.size());
      
        if (0 == iteration % 1000) {
          
          ModelSerializer.writeModel(computationGraph, "2021-06-14-bestmodel0" + iteration + "a.bin", true);
        }
      
      }
    
  }

  private void loadEarlierTrainingExamples() throws IOException, FileNotFoundException {

    try (ObjectInputStream trainExamplesInput = new ObjectInputStream(
        new FileInputStream("trainExamples.obj"))) {
      
      this.trainExamplesHistory = (List<AdversaryTrainingExample>) trainExamplesInput.readObject();
    
    } catch (ClassNotFoundException e) {
      e.printStackTrace();
    }
  }
  
  ComputationGraph performTraining(ComputationGraph computationGraph, List<AdversaryTrainingExample> trainingExamples) {

    INDArray inputBoards = Nd4j.zeros(trainingExamples.size(), 3, 3, 3);
    INDArray probabilitiesLabels = Nd4j.zeros(trainingExamples.size(), TicTacToeConstants.COLUMN_COUNT);
    INDArray valueLabels =  Nd4j.zeros(trainingExamples.size(), 1);    
    
    for (int exampleNumber = 0; exampleNumber < trainingExamples.size(); exampleNumber++) {
      
      AdversaryTrainingExample currentTrainingExample = trainingExamples.get(exampleNumber);
      inputBoards.putRow(exampleNumber, currentTrainingExample.getBoard());
      
      probabilitiesLabels.putRow(exampleNumber, currentTrainingExample.getActionIndexProbabilities());
      valueLabels.putRow(exampleNumber, Nd4j.create(1).putScalar(0, currentTrainingExample.getCurrentPlayerValue()));
    }

    MultiDataSet dataSet = new org.nd4j.linalg.dataset.MultiDataSet(new INDArray[] {inputBoards}, new INDArray[] {probabilitiesLabels, valueLabels});
    
    computationGraph.fit(dataSet);
    
    return computationGraph;
  }

  protected void evaluateOpeningAnswers(ComputationGraph convolutionalNetwork) {

    INDArray centerFieldOpeningAnswer = convolutionalNetwork.output(generateCenterFieldInputImagesConvolutional())[0];
    INDArray cornerFieldOpeningAnswer = convolutionalNetwork
        .output(generateLastCornerFieldInputImagesConvolutional())[0];
    INDArray fieldOneOpeningAnswer = convolutionalNetwork
        .output(generateFieldOneInputImagesConvolutional())[0];
    INDArray fieldOneCenterTwoOpeningAnswer = convolutionalNetwork
        .output(generateFieldOneCenterAndTwoThreatConvolutional())[0];

    log.info("Answer to center field opening: {}", centerFieldOpeningAnswer);
    log.info("Answer to last corner field opening: {}", cornerFieldOpeningAnswer);
    log.info("Answer to field one, center and two threat: {}", fieldOneCenterTwoOpeningAnswer);
    log.info("Answer to field one opening: {}", fieldOneOpeningAnswer);
  }

  INDArray generateCenterFieldInputImagesConvolutional() {

    INDArray middleFieldMove = TicTacToeConstants.EMPTY_CONVOLUTIONAL_PLAYGROUND.dup();
    INDArray emptyImage1 = Nd4j.ones(1, IMAGE_SIZE, IMAGE_SIZE).mul(-1);
    middleFieldMove.putRow(0, emptyImage1);
    middleFieldMove.putScalar(1, 1, 1, OCCUPIED_IMAGE_POINT);
    INDArray graphSingleBatchInput1 = Nd4j.create(1, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE);
    graphSingleBatchInput1.putRow(0, middleFieldMove);
    return graphSingleBatchInput1;
  }

  INDArray generateLastCornerFieldInputImagesConvolutional() {

    INDArray cornerFieldMove = TicTacToeConstants.EMPTY_CONVOLUTIONAL_PLAYGROUND.dup();
    INDArray emptyImage2 = Nd4j.ones(1, IMAGE_SIZE, IMAGE_SIZE).mul(-1);
    cornerFieldMove.putRow(0, emptyImage2);
    cornerFieldMove.putScalar(1, 2, 2, OCCUPIED_IMAGE_POINT);
    INDArray graphSingleBatchInput2 = Nd4j.create(1, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE);
    graphSingleBatchInput2.putRow(0, cornerFieldMove);
    return graphSingleBatchInput2;
  }

  INDArray generateFieldOneInputImagesConvolutional() {

    INDArray fieldOneMaxMove = TicTacToeConstants.EMPTY_CONVOLUTIONAL_PLAYGROUND.dup();
    INDArray emptyImage1 = Nd4j.ones(1, IMAGE_SIZE, IMAGE_SIZE).mul(-1);
    fieldOneMaxMove.putRow(0, emptyImage1);
    fieldOneMaxMove.putScalar(1, 0, 0, OCCUPIED_IMAGE_POINT);
    INDArray graphSingleBatchInput2 = Nd4j.create(1, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE);
    graphSingleBatchInput2.putRow(0, fieldOneMaxMove);
    return graphSingleBatchInput2;
  }

  INDArray generateFieldOneCenterAndTwoThreatConvolutional() {

    INDArray fieldOneCenterTwoMoves = TicTacToeConstants.EMPTY_CONVOLUTIONAL_PLAYGROUND.dup();
    INDArray emptyImage1 = Nd4j.ones(1, IMAGE_SIZE, IMAGE_SIZE).mul(-1);
    fieldOneCenterTwoMoves.putRow(0, emptyImage1);
    fieldOneCenterTwoMoves.putScalar(1, 0, 0, OCCUPIED_IMAGE_POINT);
    fieldOneCenterTwoMoves.putScalar(2, 1, 1, OCCUPIED_IMAGE_POINT);
    fieldOneCenterTwoMoves.putScalar(1, 0, 1, OCCUPIED_IMAGE_POINT);
    INDArray graphSingleBatchInput2 = Nd4j.create(1, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE);
    graphSingleBatchInput2.putRow(0, fieldOneCenterTwoMoves);
    return graphSingleBatchInput2;
  }
}
