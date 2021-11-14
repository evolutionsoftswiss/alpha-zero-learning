package ch.evolutionsoft.rl.netupdate;

import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.util.NetworkUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.http.client.reactive.ReactorClientHttpConnector;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.client.WebClientRequestException;

import ch.evolutionsoft.rl.AdversaryLearningConfiguration;
import ch.evolutionsoft.rl.AdversaryLearningSharedHelper;
import ch.evolutionsoft.rl.AdversaryTrainingExample;
import ch.evolutionsoft.rl.GraphLoader;

@Component
@ComponentScan("ch.evolutionsoft.rl")
public class NeuralNetUpdater {

  public static final Logger log = LoggerFactory.getLogger(NeuralNetUpdater.class);
  
  public static final String CONTROLLER_BASE_URL = "http://localhost:8080/alpha-zero";
  
  public static final int ONE_GIGA_BYTE = 1024 * 1024 * 1024;

  AdversaryLearningConfiguration adversaryLearningConfiguration;
  
  AdversaryLearningSharedHelper adversaryLearningSharedHelper;

  WebClient webClient;

  final ParameterizedTypeReference<List<AdversaryTrainingExample>> paramterizedTypeReference =
      new ParameterizedTypeReference<List<AdversaryTrainingExample>>() {
    
  };

  public NeuralNetUpdater() {
    
    this.webClient = WebClient.builder().
        clientConnector(new ReactorClientHttpConnector()).
        codecs(configurer -> configurer.defaultCodecs().maxInMemorySize(ONE_GIGA_BYTE)).
        build();
  }

  public static void main(String[] args) throws IOException {

    AnnotationConfigApplicationContext applicationContext =
       new AnnotationConfigApplicationContext(NeuralNetUpdater.class);
    
    NeuralNetUpdater neuralNetUpdater = applicationContext.getBean(NeuralNetUpdater.class);
    
    neuralNetUpdater.initialize();
    neuralNetUpdater.listenForNewTrainingExamples();
    
    applicationContext.close();
  }

  public void initialize() throws IOException {

    while (!this.adversaryLearningIsReady()) {
      // Wait for startup
    }

    this.adversaryLearningConfiguration = 
        this.webClient.
        get().
        uri(URI.create(CONTROLLER_BASE_URL + "/adversaryLearningConfiguration")).
        retrieve().
        bodyToMono(AdversaryLearningConfiguration.class).
        block();
       
    
    this.adversaryLearningSharedHelper = new AdversaryLearningSharedHelper(
        adversaryLearningConfiguration);

    if (null != adversaryLearningConfiguration &&
        adversaryLearningConfiguration.getIterationStart() > 1) {
      this.adversaryLearningSharedHelper.loadEarlierTrainingExamples(true);
    }
    
    while (!getAdversaryLearningIsInitialized()) {
      // Wait for initialization
    }
    
    log.info("Neural Net Updater initialized");
  }
  
  public void listenForNewTrainingExamples() {

    final String targetUrl = CONTROLLER_BASE_URL + "/newTrainingExamples";
    
    for (int iteration = adversaryLearningConfiguration.getIterationStart();
        iteration < adversaryLearningConfiguration.getIterationStart() + 
        adversaryLearningConfiguration.getNumberOfIterations();
        iteration++) {
        
      List<AdversaryTrainingExample> newExamples = 
            webClient.get().
            uri(URI.create(targetUrl)).
            retrieve().
            bodyToMono(paramterizedTypeReference).
            block();
        ComputationGraph computationGraph = fitNeuralNet(newExamples);
        try {
          
          ModelSerializer.writeModel(computationGraph, adversaryLearningConfiguration.getBestModelFileName(), true);

          webClient.
          put().
          uri(URI.create(CONTROLLER_BASE_URL + "/modelUpdated")).
          retrieve().
          bodyToMono(Void.class).
          block();
        
        } catch (WebClientRequestException wcre) {

          log.warn("Continue next iteration, current iteration {} failed, "
              + "encountered ResourceAccessException", iteration);

        } catch (IOException ioe) {

          log.error("Error writing updated model", ioe);
        }
    }
  }

  boolean adversaryLearningIsReady() {
    
    try {
      getAdversaryLearningIsInitialized();
      return true;
    
    } catch (WebClientRequestException wcre) {
      
      return false;
    }
  }

  boolean getAdversaryLearningIsInitialized() {

    return this.webClient.
        get().
        uri(URI.create(CONTROLLER_BASE_URL + "/isInitialized")).
        retrieve().
        bodyToMono(Boolean.class).
        block();
  }
  
  ComputationGraph fitNeuralNet(List<AdversaryTrainingExample> newExamples) {
    
    this.adversaryLearningSharedHelper.replaceOldTrainingExamplesWithNewActionProbabilities(newExamples);
    this.adversaryLearningSharedHelper.resizeTrainExamplesHistory();

    List<AdversaryTrainingExample> trainingExamples =
        new ArrayList<>(this.adversaryLearningSharedHelper.getTrainExamplesHistory().values());
    Collections.shuffle(trainingExamples);
    
    ComputationGraph computationGraph = GraphLoader.loadComputationGraph(adversaryLearningConfiguration);
    
    int batchSize = this.adversaryLearningConfiguration.getBatchSize();
    long trainingExamplesSize = trainingExamples.size();
    int batchNumber = (int) (1 + trainingExamplesSize / batchSize);
    
    List<MultiDataSet> batchedMultiDataSet = createMiniBatchList(trainingExamples);

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
      List<AdversaryTrainingExample> trainingExamples) {
 
    int batchSize = adversaryLearningConfiguration.getBatchSize();
    int trainingExamplesSize = trainingExamples.size();
    int batchNumber = 1 + trainingExamplesSize / batchSize;
    if (0 == trainingExamplesSize % batchSize) {
      batchNumber--;
    }
 
    long[] gameInputBoardStackShape = trainingExamples.get(0).getBoard().shape();
    
    List<MultiDataSet> batchedMultiDataSet = new LinkedList<>();

    for (int currentBatch = 0; currentBatch < batchNumber; currentBatch++) {

      INDArray inputBoards = Nd4j.zeros(batchSize, gameInputBoardStackShape[0], gameInputBoardStackShape[1],
          gameInputBoardStackShape[2]);
      INDArray probabilitiesLabels = Nd4j.zeros(batchSize, adversaryLearningConfiguration.getNumberOfAllAvailableMoves());
      INDArray valueLabels = Nd4j.zeros(batchSize, 1);
      
      if (currentBatch >= batchNumber - 1) {

        int lastBatchSize = trainingExamplesSize % batchSize;
        inputBoards = Nd4j.zeros(lastBatchSize, gameInputBoardStackShape[0], gameInputBoardStackShape[1],
        gameInputBoardStackShape[2]);
        probabilitiesLabels = Nd4j.zeros(lastBatchSize, adversaryLearningConfiguration.getNumberOfAllAvailableMoves());
        valueLabels = Nd4j.zeros(lastBatchSize, 1);
      }

      for (int batchExample = 0, exampleNumber = currentBatch * batchSize;
          exampleNumber < (currentBatch + 1) * batchSize && exampleNumber < trainingExamplesSize;
          exampleNumber++, batchExample++) {
        
        AdversaryTrainingExample currentTrainingExample = trainingExamples.get(exampleNumber);
        inputBoards.putRow(batchExample, currentTrainingExample.getBoard());
  
        INDArray actionIndexProbabilities = Nd4j.zeros(adversaryLearningConfiguration.getNumberOfAllAvailableMoves());
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
