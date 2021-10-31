package ch.evolutionsoft.rl;

import java.io.IOException;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.util.NetworkUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class NeuralNetUpdater {

  public static final Logger log = LoggerFactory.getLogger(NeuralNetUpdater.class);
  
  AdversaryLearningController adversaryLearningController;

  AdversaryLearningConfiguration adversaryLearningConfiguration;
  
  List<AdversaryTrainingExample> adversaryTrainingExamples;
  
  Game initialGame;

  public NeuralNetUpdater(AdversaryLearningController adversaryLearningController) {

    this.adversaryLearningController = adversaryLearningController;
    this.adversaryLearningConfiguration = adversaryLearningController.getAdversaryLearningConfiguration();
    this.initialGame = adversaryLearningController.getInitialGame();
  }

  public ExecutorService listenForNewTrainingExamples() {
    
    ExecutorService executorService = Executors.newSingleThreadExecutor();
 
    for (int iteration = adversaryLearningConfiguration.getIterationStart();
        iteration < adversaryLearningConfiguration.getIterationStart() + 
        adversaryLearningConfiguration.getNumberOfIterations();
        iteration++) {
    
      executorService.execute(() -> {
        
        adversaryTrainingExamples = adversaryLearningController.getAdversaryTrainingExamples();
        ComputationGraph computationGraph = fitNeuralNet();
        try {
          
          ModelSerializer.writeModel(computationGraph, adversaryLearningConfiguration.getBestModelFileName(), true);
          adversaryLearningController.modelUpdated();
        } catch (IOException ioe) {
          log.error("Error writing updated model", ioe);
        }
      });
    }
    
    return executorService;
  }

  ComputationGraph fitNeuralNet() {

    List<AdversaryTrainingExample> trainingExamples = this.adversaryTrainingExamples;
    
    ComputationGraph computationGraph = GraphLoader.loadComputationGraph(adversaryLearningConfiguration);
    
    int batchSize = this.adversaryLearningConfiguration.getBatchSize();
    int trainingExamplesSize = trainingExamples.size();
    int batchNumber = 1 + trainingExamplesSize / batchSize;
    
    List<MultiDataSet> batchedMultiDataSet = createMiniBatchList(trainingExamples, initialGame);

    for (int batchIteration = 0; batchIteration < batchNumber; batchIteration++) {
      
      log.info("Batch size for batch number {} is {}", 
          batchIteration,
          batchedMultiDataSet.get(batchIteration).asList().size());

      computationGraph.fit(batchedMultiDataSet.get(batchIteration));
      
      log.info("Fitted model with batch number {}", batchIteration);
    }

    log.info("Iterations (number of updates) from computation graph model is {}",
        computationGraph.getIterationCount());
    log.info("Learning rate from computation graph model layer 'OutputLayer': {}",
        NetworkUtils.getLearningRate(computationGraph, "OutputLayer"));

    return computationGraph;
  }

  List<MultiDataSet> createMiniBatchList(
      List<AdversaryTrainingExample> trainingExamples,
      Game initialGame) {
 
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
