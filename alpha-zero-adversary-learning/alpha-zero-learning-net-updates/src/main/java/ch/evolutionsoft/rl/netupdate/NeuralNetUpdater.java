package ch.evolutionsoft.rl.netupdate;

import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

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

  ExecutorService initializationExecutor = Executors.newSingleThreadExecutor();

  WebClient webClient;

  final ParameterizedTypeReference<Set<AdversaryTrainingExample>> parameterizedTypeReference =
      new ParameterizedTypeReference<Set<AdversaryTrainingExample>>() {
    
  };

  public NeuralNetUpdater() {

    this.webClient = WebClient.builder().
        clientConnector(new ReactorClientHttpConnector()).
        codecs(configurer -> configurer.defaultCodecs().maxInMemorySize(ONE_GIGA_BYTE)).
        build();
  }

  public static void main(String[] args) {

    AnnotationConfigApplicationContext applicationContext =
       new AnnotationConfigApplicationContext(NeuralNetUpdater.class);
    
    NeuralNetUpdater neuralNetUpdater = applicationContext.getBean(NeuralNetUpdater.class);

    neuralNetUpdater.initializationExecutor.submit(new Callable<Void>() {

      @Override
      public Void call() throws Exception {

        neuralNetUpdater.initialize();
        
        return null;
      }
      
    });
    neuralNetUpdater.initializationExecutor.shutdown();
    
    neuralNetUpdater.listenForNewTrainingExamples();
    
    applicationContext.close();
  }

  public void initialize() throws IOException {

    log.info("Wait for AdversaryLearning readiness");
    
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
      this.adversaryLearningSharedHelper.loadEarlierTrainingExamples();
    }
    
    while (!getAdversaryLearningIsInitialized()) {
      // Wait for initialization
    }
    
    log.info("Neural Net Updater initialized");
  }
  
  public void listenForNewTrainingExamples() {

    String targetUrl = CONTROLLER_BASE_URL + "/newTrainingExamples";
    Set<AdversaryTrainingExample> newExamples = null;
    Set<AdversaryTrainingExample> previousExamples = null;
    
    ExecutorService netUpdaterExecutor = null;
    
    boolean firstUpdate = true;

    try {
      
      while (!this.initializationExecutor.awaitTermination(Long.MAX_VALUE, TimeUnit.MINUTES)) {
        // Wait for initilzation
      }
      
      for (int iteration = adversaryLearningConfiguration.getIterationStart() - 1;
          iteration < adversaryLearningConfiguration.getIterationStart() + 
          adversaryLearningConfiguration.getNumberOfIterations() - 1;
          iteration++) {
          
        previousExamples = newExamples;
        newExamples = 
              webClient.get().
              uri(URI.create(targetUrl)).
              retrieve().
              bodyToMono(parameterizedTypeReference).
              block();
        
        if (!firstUpdate) {
          while (null != netUpdaterExecutor && !netUpdaterExecutor.awaitTermination(Long.MAX_VALUE, TimeUnit.MINUTES)) {
            // Wait for previous net update
          }
          
          final int updateIteration = iteration;
          netUpdaterExecutor = Executors.newSingleThreadExecutor();
          final List<AdversaryTrainingExample> finalInputList = new LinkedList<>(previousExamples);
          netUpdaterExecutor.execute(() -> {
    
              ComputationGraph computationGraph = fitNeuralNet(finalInputList, updateIteration);
    
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
                      + "encountered ResourceAccessException", updateIteration);
    
                } catch (IOException ioe) {
    
                  log.error("Error writing updated model", ioe);
                }
            }
          );

          netUpdaterExecutor.shutdown();
        }
        firstUpdate = false;
      }
    } catch (InterruptedException ie) {
      
      Thread.currentThread().interrupt();
      throw new NeuralNetUpdaterRuntimeException(ie);
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
  
  ComputationGraph fitNeuralNet(List<AdversaryTrainingExample> newExamples, int updateIteration) {
    
    log.info("Replace old present examples with new ones");
    this.adversaryLearningSharedHelper.replaceOldTrainingExamplesWithNewActionProbabilities(
        newExamples);

    log.info("Resize train examples history");
    this.adversaryLearningSharedHelper.resizeTrainExamplesHistory(updateIteration);

    List<AdversaryTrainingExample> trainingExamples =
        new ArrayList<>(this.adversaryLearningSharedHelper.getTrainExamplesHistory().values());
    Collections.shuffle(trainingExamples);
    
    ComputationGraph computationGraph = GraphLoader.loadComputationGraph(adversaryLearningConfiguration);
    
    int batchSize = this.adversaryLearningConfiguration.getBatchSize();
    long trainingExamplesSize = trainingExamples.size();
    int batchNumber = (int) (1 + trainingExamplesSize / batchSize);

    log.info("Create minibatches");    
    List<MultiDataSet> batchedMultiDataSet = createMiniBatchList(trainingExamples);

    for (int updateCycle = 1; updateCycle <= 2; updateCycle++) {
      for (int batchIteration = 0; batchIteration < batchNumber; batchIteration++) {
          
        log.info("Batch size for batch number {} is {}", 
            batchIteration,
            batchedMultiDataSet.get(batchIteration).asList().size());
    
        computationGraph.fit(batchedMultiDataSet.get(batchIteration));
          
        log.info("Fitted model with batch number {}", batchIteration);
      }
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
  
        probabilitiesLabels.putRow(batchExample, currentTrainingExample.getActionIndexProbabilities());
  
        valueLabels.putRow(batchExample, Nd4j.zeros(1).putScalar(0, currentTrainingExample.getCurrentPlayerValue()));
      }
      
      batchedMultiDataSet.add( new org.nd4j.linalg.dataset.MultiDataSet(new INDArray[] { inputBoards },
      new INDArray[] { probabilitiesLabels, valueLabels }));
    }
    
    return batchedMultiDataSet;
  }
}
