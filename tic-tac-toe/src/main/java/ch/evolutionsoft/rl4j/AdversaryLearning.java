package ch.evolutionsoft.rl4j;

import static ch.evolutionsoft.net.game.tictactoe.TicTacToeConstants.EMPTY_IMAGE_POINT;
import static ch.evolutionsoft.net.game.tictactoe.TicTacToeConstants.IMAGE_CHANNELS;
import static ch.evolutionsoft.net.game.tictactoe.TicTacToeConstants.IMAGE_SIZE;
import static ch.evolutionsoft.net.game.tictactoe.TicTacToeConstants.OCCUPIED_IMAGE_POINT;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ch.evolutionsoft.net.game.tictactoe.TicTacToeConstants;
import ch.evolutionsoft.rl4j.tictactoe.EvaluationMain;
import ch.evolutionsoft.rl4j.tictactoe.TicTacToe;

public class AdversaryLearning {

  private static final Logger log = LoggerFactory.getLogger(AdversaryLearning.class);

  List<AdversaryTrainingExample> trainExamplesHistory = new ArrayList<>();
  
  ComputationGraph computationGraph;
  ComputationGraph pComputationGraph;
  
  int numberOfEpisodes = 30;
  
  int currentPlayer = TicTacToeConstants.MAX_PLAYER_CHANNEL;
  int episodeStepsTemperatureThreshold = 20;
  
  int maxTrainExamplesHistory = 20000;
  
  MonteCarloTreeSearch mcts;

  public AdversaryLearning(ComputationGraph computationGraph, int numberOfEpisodes) {
    
    this.computationGraph = computationGraph;
    this.numberOfEpisodes = numberOfEpisodes;
  }
  
  List<AdversaryTrainingExample> executeEpisode() {
    
    List<AdversaryTrainingExample> trainExamples = new ArrayList<>();
    
    INDArray currentBoard = TicTacToeConstants.EMPTY_CONVOLUTIONAL_PLAYGROUND;
    int episodeStep = 0;
    
    while (!TicTacToe.gameEnded(currentBoard)) {
      
      episodeStep++;
      float temperature = episodeStep < episodeStepsTemperatureThreshold ? 1 : 0;
      INDArray actionProbabilities = this.mcts.getActionValues(currentBoard, temperature);
      
      AdversaryTrainingExample trainingExample = new AdversaryTrainingExample(currentBoard, currentPlayer, actionProbabilities);
      
      if (trainExamples.contains(trainingExample)) {
        
        trainExamples.remove(trainingExample);
      }
      
      trainExamples.add(trainingExample);
      
      List<AdversaryTrainingExample> symmetries = TicTacToe.getSymmetries(currentBoard.dup(), actionProbabilities.dup(), currentPlayer);
      
      for (AdversaryTrainingExample symmetryExample : symmetries) {
        
        if (!trainExamples.contains(symmetryExample)) {
          
          trainExamples.add(symmetryExample);
        }
      }
      
      
      DistributedRandomNumberGenerator d = new DistributedRandomNumberGenerator(actionProbabilities);
      int moveAction = d.getDistributedRandomNumber();
 
      currentBoard = TicTacToe.makeMove(currentBoard, moveAction, currentPlayer);
      
      if (TicTacToe.gameEnded(currentBoard)) {
        
        int gameResult = 0;
        if (TicTacToe.hasWon(currentBoard, currentPlayer)) {
          
          gameResult = 1;
        }
        
        for (AdversaryTrainingExample trainExample : trainExamples) {
          
          trainExample.setCurrentPlayerValue(trainExample.getCurrentPlayer() == currentPlayer ? gameResult : -gameResult);
        }
        
        return trainExamples;
      }
      
      currentPlayer = currentPlayer == TicTacToeConstants.MAX_PLAYER_CHANNEL ?
          TicTacToeConstants.MIN_PLAYER_CHANNEL : TicTacToeConstants.MAX_PLAYER_CHANNEL;
    }
    
    return null;
  }
  
  void performLearning() throws IOException {

    if (true) {
      
      this.pComputationGraph = ModelSerializer.restoreComputationGraph("src/main/resources/tempmodel.bin", true);
      this.computationGraph = ModelSerializer.restoreComputationGraph("src/main/resources/bestmodel.bin", true);
      
      /*try (ObjectInputStream trainExamplesInput = new ObjectInputStream(
          new FileInputStream("src/main/resources/trainExamples.obj"))) {
        
        this.trainExamplesHistory = (List<AdversaryTrainingExample>) trainExamplesInput.readObject();
      
      } catch (ClassNotFoundException e) {
        e.printStackTrace();
      }*/
    }
    
    for (int iteration = 0; iteration < 100; iteration++) {
    
      for (int episodeNumber = 1; episodeNumber <= this.numberOfEpisodes; episodeNumber++) {
          
        this.mcts = new MonteCarloTreeSearch(computationGraph);
          
        List<AdversaryTrainingExample> newExamples = this.executeEpisode();
          
        for (AdversaryTrainingExample trainExample : newExamples) {
            
          if (this.trainExamplesHistory.contains(trainExample)) {
              
            this.trainExamplesHistory.remove(trainExample);
          }
            
          this.trainExamplesHistory.add(trainExample);
        }
      }
    
      while (this.trainExamplesHistory.size() > this.maxTrainExamplesHistory) {
        this.trainExamplesHistory.remove(0);
      }
      
      /*try (ObjectOutputStream trainExamplesOutput = new ObjectOutputStream(
          new FileOutputStream("src/main/resources/trainExamples.obj"))) {
        // TODO write readable train examples history
        trainExamplesOutput.writeObject(trainExamplesHistory);
      }*/
      
      List<AdversaryTrainingExample> trainExamples = new ArrayList<>(this.trainExamplesHistory);
      Collections.shuffle(trainExamples);
      
      ModelSerializer.writeModel(computationGraph, "src/main/resources/tempmodel.bin", true);
      this.pComputationGraph = ModelSerializer.restoreComputationGraph("src/main/resources/tempmodel.bin", true);
      
      MonteCarloTreeSearch pMcts = new MonteCarloTreeSearch(this.pComputationGraph);

      if (iteration > 1) {
        this.computationGraph = this.performTraining(this.computationGraph, trainExamples);
      }
      MonteCarloTreeSearch nMcts = new MonteCarloTreeSearch(this.computationGraph);
      
      AdversaryAgentDriver adversaryAgentDriver = new AdversaryAgentDriver(pMcts, nMcts);
      int[] gameResults = adversaryAgentDriver.playGames(30, TicTacToeConstants.EMPTY_CONVOLUTIONAL_PLAYGROUND);
      
      log.info("New model wins {} / prev model wins {} / draws {}", gameResults[1], gameResults[0], gameResults[2]);
      
      if ( (gameResults[1] + 0.1 * gameResults[2]) / (float) (gameResults[0] + gameResults[1] + 0.1 * gameResults[2]) < 0.55) {
        log.info("Rejecting new model");
        this.computationGraph = ModelSerializer.restoreComputationGraph("src/main/resources/tempmodel.bin", true);
      
      } else {
        
        log.info("Accepting new model");
        ModelSerializer.writeModel(computationGraph, "src/main/resources/checkpoint" + iteration + ".bin", true);
        ModelSerializer.writeModel(computationGraph, "src/main/resources/bestmodel.bin", true);
        evaluateOpeningAnswers(pComputationGraph);
        evaluateOpeningAnswers(computationGraph);
        EvaluationMain.evaluateNetwork(computationGraph);
      }
      
      log.info("Iteration {} ended, train examples {}", iteration, this.trainExamplesHistory.size());
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
    INDArray fieldSixCenterOpeningAnswer = convolutionalNetwork
        .output(generateFieldSixAndCenterInputImagesConvolutional())[0];

    log.info("Answer to center field opening: {}", centerFieldOpeningAnswer);
    log.info("Answer to last corner field opening: {}", cornerFieldOpeningAnswer);
    log.info("Answer to field six and center response opening: {}", fieldSixCenterOpeningAnswer);
  }

  INDArray generateCenterFieldInputImagesConvolutional() {

    INDArray middleFieldMove = TicTacToeConstants.EMPTY_CONVOLUTIONAL_PLAYGROUND.dup();
    INDArray emptyImage1 = Nd4j.ones(1, IMAGE_SIZE, IMAGE_SIZE);
    emptyImage1.putScalar(0, 1, 1, EMPTY_IMAGE_POINT);
    middleFieldMove.putRow(0, emptyImage1);
    middleFieldMove.putScalar(1, 1, 1, OCCUPIED_IMAGE_POINT);
    INDArray graphSingleBatchInput1 = Nd4j.create(1, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE);
    graphSingleBatchInput1.putRow(0, middleFieldMove);
    return graphSingleBatchInput1;
  }

  INDArray generateLastCornerFieldInputImagesConvolutional() {

    INDArray cornerFieldMove = TicTacToeConstants.EMPTY_CONVOLUTIONAL_PLAYGROUND.dup();
    INDArray emptyImage2 = Nd4j.ones(1, IMAGE_SIZE, IMAGE_SIZE);
    emptyImage2.putScalar(0, 2, 2, EMPTY_IMAGE_POINT);
    cornerFieldMove.putRow(0, emptyImage2);
    cornerFieldMove.putScalar(1, 2, 2, OCCUPIED_IMAGE_POINT);
    INDArray graphSingleBatchInput2 = Nd4j.create(1, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE);
    graphSingleBatchInput2.putRow(0, cornerFieldMove);
    return graphSingleBatchInput2;
  }

  INDArray generateFieldSixAndCenterInputImagesConvolutional() {

    INDArray fieldSixMaxCenterMinMove = TicTacToeConstants.EMPTY_CONVOLUTIONAL_PLAYGROUND.dup();
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
