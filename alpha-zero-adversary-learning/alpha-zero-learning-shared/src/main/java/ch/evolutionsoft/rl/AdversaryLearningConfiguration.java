package ch.evolutionsoft.rl;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;

import org.apache.commons.lang3.StringUtils;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonSetter;

/**
 * {@link AdversaryLearningConfiguration} defines several configuration parameters
 * affecting the behavior of alpha zero learning.
 * 
 * @author evolutionsoft
 */
public class AdversaryLearningConfiguration {
  
  public static final String OBJECT_ENDING = ".obj";
  
  /**
   * A learningRateSchedule defining different learning rates in function of the
   * number of performed net iterations meaning calls to {@link ComputationGraph} fit method here.
   * 
   * The iterations defining the learning rate for ISchedule is only directly related to alpha zero 
   * numberOfIterations with a batch size that covers all existing examples.
   * 
   * Otherwise with mini batches > 1 it is dependent of 
   * performed {@link ComputationGraph} updates using the fit method.
   */
  private MapSchedule learningRateSchedule;

  /**
   * Size of mini batches used to perform {@link ComputationGraph} updates with fit.
   *
   * TicTacToe currently uses a value greater than all possible {@link AdversaryTrainingExample}
   * numbers leading to use one single batch always.
   */
  private int batchSize;

  /**
   * Value of the dirichlet alpha used to add noise to move probability distributions.
   * TicTacToe and ConnectFour use a greater value nearer to one compared to other games known values from Alpha Zero.
   */
  private double dirichletAlpha;

  /**
   * Weight of the dirichlet noise added to currently known move probabilities.
   */
  private double dirichletWeight;
  
  /**
   * The number of all available moves define some shapes and capacity initializations.
   */
  private int numberOfAllAvailableMoves;

  /**
   * numberOfEpisodesBeforePotentialUpdate stands for numberOfEpisodes.
   * An Alpha Zero episode is one game from start to end. Each episode generates potentially new
   * training examples used to train the neural net. numberOfEpisodesBeforePotentialUpdate defines how much times a game 
   * will be run from start to end to gather training examples before a potential neural net update.
   */
  private int numberOfEpisodesBeforePotentialUpdate;

  /**
   * To execute episodes with the same neural net model in parallel, it should be most effective
   * to use a multiple numberOfEpisodesBeforePotentialUpdate of numberOfEpisodeThreads.
   * Default is half the available processors. If you've got as many or more cpu cores than the
   * numberOfEpisodesBeforePotentialUpdate, you should use the same as numberOfEpisodesBeforePotentialUpdate.
   * That would be most effective for gathering {@link AdversaryTrainingExample} in different epsiodes.
   */
  private int numberOfEpisodeThreads;
  
  /**
   * Used to continue training after program termination.
   */
  private boolean continueTraining;

  /**
   * initialIteration should not be set manually, default is 1 and with continueTraining = true
   * the value is taken from the latest stored iteration from trainExamples files.
   */
  private int initialIteration;

  /**
   * numberOfIterations here means the total number of Alpha Zero iterations.
   */
  private int numberOfIterations;

  /**
   * After checkPointIterationsFrequency store additional files containing the current
   * model and training examples.
   */
  private int checkPointIterationsFrequency;

  /**
   * When the temperature used in {@link MonteCarloTreeSearch} getActionValues() should become 0.
   * Currently only 1 or 0 are used. With too small values > 0 overflows were observed.
   * A temperature == 0 will lead to move action probabilities all zero, expect
   * one being one. Temperatures > 0 keep probabilities > 0 for all move actions
   * in function of the number of visits during {@link MonteCarloTreeSearch}.
   */
  private int fromNumberOfIterationsReducedTemperature;
  
  /**
   * Currently used approach in TicTacToe example implementation.
   * Also in early iterations use zero temperature after having reached the
   * specified number of moves in an alpha zero iteration.
   */
  private int fromNumberOfMovesReducedTemperature;

  /**
   * Use another value than zero for reduced temperature
   */
  private double reducedTemperature;
  
  /**
   * The maximum number of train examples to keep in history and reuse for neural net fit.
   * TicTacToe never exceeds the used value of 5000. ConnectFour uses 80'000 and removes
   * early {@link AdversaryTrainingExample} from around 300 iterations on.
   * Typical values for Go 19x19 are 500'000 to 2 million eaxample positions.
   */
  private int maxTrainExamplesHistory;

  /**
   * 
   */
  private int maxTrainExamplesHistoryFromIteration;

  /**
   * {@link MonteCarloTreeSearch} parameter influencing exploration / exploitation of
   * different move actions. TicTacToe uses 1.5. ConnectFour uses 2.5.
   */
  private double uctConstantFactor;

  /**
   * How much single playout steps should {@link MonteCarloTreeSearch} perform.
   * TicTacToe example implementation uses 30. ConnectFour uses 200.
   * Typical values for Go 9x9 and Go 19x19 would be 400 and 1600.
   */
  private int numberOfMonteCarloSimulations;

  /**
   * The file name and extension without path to use for the current best model.
   */
  private String bestModelFileName;

  /**
   * The file name and extension without path to use for storing the generated
   * {@link AdversaryTrainingExample} during self play.
   */
  private String trainExamplesFileName;

  /**
   * Default initial values for TicTacToe example implementation.
   * 
   * @author evolutionsoft
   */
  public static class Builder {

    private MapSchedule learningRateSchedule;
    private int batchSize = 8192;

    private double dirichletAlpha = 0.8;
    private double dirichletWeight = 0.40;
    private int numberOfAllAvailableMoves = 9;
    private int numberOfEpisodesBeforePotentialUpdate = 10;
    private int numberOfEpisodeThreads = Runtime.getRuntime().availableProcessors() / 2;
    private boolean continueTraining = false;
    private int initialIteration = 1;
    private int numberOfIterations = 250;
    private int checkPointIterationsFrequency = 50;
    private int fromNumberOfIterationsReducedTemperature = -1;
    private int fromNumberOfMovesReducedTemperature = -1;
    private double reducedTemperature = 0;
    private int maxTrainExamplesHistory = 5000;
    private int maxTrainExamplesHistoryFromIteration = -1;

    private String bestModelFileName = "model.bin";
    private String trainExamplesFileName = "trainExamples.obj";

    private double uctConstantFactor = 1.5;
    private int numberOfMonteCarloSimulations = 25;
    
    public AdversaryLearningConfiguration build() {
      
      AdversaryLearningConfiguration configuration = new AdversaryLearningConfiguration();
      
      configuration.learningRateSchedule = learningRateSchedule;
      configuration.batchSize = batchSize;
      configuration.dirichletAlpha = dirichletAlpha;
      configuration.dirichletWeight = dirichletWeight;
      configuration.numberOfAllAvailableMoves = numberOfAllAvailableMoves;
      configuration.numberOfEpisodesBeforePotentialUpdate = numberOfEpisodesBeforePotentialUpdate;
      configuration.numberOfEpisodeThreads = numberOfEpisodeThreads;
      configuration.continueTraining = continueTraining;
      configuration.initialIteration = initialIteration;
      configuration.numberOfIterations = numberOfIterations;
      configuration.checkPointIterationsFrequency = checkPointIterationsFrequency;
      configuration.fromNumberOfIterationsReducedTemperature = fromNumberOfIterationsReducedTemperature;
      configuration.fromNumberOfMovesReducedTemperature = fromNumberOfMovesReducedTemperature;
      configuration.reducedTemperature = reducedTemperature;
      configuration.maxTrainExamplesHistory = maxTrainExamplesHistory;
      configuration.maxTrainExamplesHistoryFromIteration = maxTrainExamplesHistoryFromIteration;
      configuration.uctConstantFactor = uctConstantFactor;
      configuration.numberOfMonteCarloSimulations = numberOfMonteCarloSimulations;
      configuration.bestModelFileName = getAbsolutePathFrom(bestModelFileName);
      configuration.trainExamplesFileName = getAbsolutePathFrom(trainExamplesFileName);
      
      return configuration;
    }

    public Builder learningRateSchedule(MapSchedule learningRateSchedule) {
      this.learningRateSchedule = learningRateSchedule;
      return this;
    }
    
    public Builder batchSize(int batchSize) {
      this.batchSize = batchSize;
      return this;
    }

    public Builder dirichletAlpha(double dirichletAlpha) {
      this.dirichletAlpha = dirichletAlpha;
      return this;
    }

    public Builder dirichletWeight(double dirichletWeight) {
      this.dirichletWeight = dirichletWeight;
      return this;
    }

    public Builder fromNumberOfIterationsReducedTemperature(int fromNumberOfIterationsReducedTemperature) {
      this.fromNumberOfIterationsReducedTemperature = fromNumberOfIterationsReducedTemperature;
      return this;
    }

    public Builder fromNumberOfMovesReducedTemperature(int fromNumberOfMovesReducedTemperature) {
      this.fromNumberOfMovesReducedTemperature = fromNumberOfMovesReducedTemperature;
      return this;
    }
    
    public Builder reducedTemperature(double temperature) {
      this.reducedTemperature = temperature;
      return this;
    }
    
    public Builder numberOfAllAvailableMoves(int numberOfAllAvailableMoves) {
      this.numberOfAllAvailableMoves = numberOfAllAvailableMoves;
      return this;
    }

    public Builder numberOfEpisodesBeforePotentialUpdate(int numberOfEpisodesBeforePotentialUpdate) {
      this.numberOfEpisodesBeforePotentialUpdate = numberOfEpisodesBeforePotentialUpdate;
      return this;
    }
    
    public Builder numberOfEpisodeThreads(int numberOfEpisodeThreads) {
      this.numberOfEpisodeThreads = numberOfEpisodeThreads;
      return this;
    }
    
    public Builder continueTraining(boolean continueTraining) {
      this.continueTraining = continueTraining;
      return this;
    }

    public Builder initialIteration(int initialIteration) {
      this.initialIteration = initialIteration;
      return this;
    }

    public Builder numberOfIterations(int totalNumberOfIterations) {
      this.numberOfIterations = totalNumberOfIterations;
      return this;
    }

    public Builder checkPointIterationsFrequency(int checkPointIterationsFrequency) {
      
      this.checkPointIterationsFrequency = checkPointIterationsFrequency;
      return this;
    }
    
    public Builder maxTrainExamplesHistory(int maxTrainExamplesHistory) {
      this.maxTrainExamplesHistory = maxTrainExamplesHistory;
      return this;
    }
    
    public Builder maxTrainExamplesHistoryFromIteration(int maxTrainExamplesHistoryFromIteration) {
      this.maxTrainExamplesHistoryFromIteration = maxTrainExamplesHistoryFromIteration;
      return this;
    }

    public Builder uctConstantFactor(double uctConstantFactor) {
      this.uctConstantFactor = uctConstantFactor;
      return this;
    }

    public Builder numberOfMonteCarloSimulations(int numberOfMonteCarloSimulations) {
      this.numberOfMonteCarloSimulations = numberOfMonteCarloSimulations;
      return this;
    }
    
    public Builder bestModelFileName(String bestModelFileName) {
      this.bestModelFileName = bestModelFileName;
      return this;
    }
    
    public Builder trainExamplesFileName(String trainExamplesFileName) {
      this.trainExamplesFileName = trainExamplesFileName;
      return this;
    }
  }
  
  public String toString() {
    
    return " learningRate: " + this.learningRateSchedule  +
        "\n batch size: " + this.batchSize +
        "\n dirichletAlpha: " + this.dirichletAlpha + 
        "\n dirichletWeight: " + this.dirichletWeight +
        "\n numberOfAllAvailableMoves: " + this.numberOfAllAvailableMoves +
        "\n numberOfEpisodesBeforePotentialUpdate: " + this.numberOfEpisodesBeforePotentialUpdate + 
        "\n numberOfEpisodeThreads: " + this.numberOfEpisodeThreads +
        "\n continueTraining: " + this.continueTraining + 
        "\n initialIteration: " + this.initialIteration + 
        "\n numberOfIterations: " + this.numberOfIterations +
        "\n checkPointIterationsFrequency: " + this.checkPointIterationsFrequency +
        "\n fromNumberOfIterationsReducedTemperature: " + this.fromNumberOfIterationsReducedTemperature +
        "\n fromNumberOfMovesReducedTemperature: " + this.fromNumberOfMovesReducedTemperature +
        "\n reducedTemperature: " + this.reducedTemperature +
        "\n maxTrainExamplesHistory: " + this.maxTrainExamplesHistory +
        "\n maxTrainExamplesHistoryFromIteration: " + this.maxTrainExamplesHistoryFromIteration +
        "\n currentMaxTrainExamplesHistory: " + this.getCurrentMaxTrainExamplesHistory(initialIteration) +
        "\n cpUct: " + this.uctConstantFactor +
        "\n numberOfMonteCarloSimulations: " + this.numberOfMonteCarloSimulations +
        "\n bestModelFileName: " + getAbsolutePathFrom(this.bestModelFileName) +
        "\n trainExamplesFileNames: " + getAbsolutePathFrom(this.trainExamplesFileName);
  }
  
  public MapSchedule getLearningRateSchedule() {
    return learningRateSchedule;
  }

  @JsonProperty("learningRateSchedule")
  public Map<Integer, Double> getLearningRateScheduleMap() {
    
    return this.learningRateSchedule.getValues();
    
  }

  public void setLearningRateSchedule(MapSchedule learningRateSchedule) {
    this.learningRateSchedule = learningRateSchedule;
  }

  @JsonSetter
  public void setLearningRateSchedule(Map<Integer, Double> learningRatesByIterations) {
    this.learningRateSchedule = new MapSchedule(ScheduleType.ITERATION, learningRatesByIterations);
  }
  
  public int getBatchSize() {
    return batchSize;
  }

  public void setBatchSize(int batchSize) {
    this.batchSize = batchSize;
  }

  public double getDirichletAlpha() {
    return dirichletAlpha;
  }

  public void setDirichletAlpha(double dirichletAlpha) {
    this.dirichletAlpha = dirichletAlpha;
  }

  public double getDirichletWeight() {
    return dirichletWeight;
  }

  public void setDirichletWeight(double dirichletWeight) {
    this.dirichletWeight = dirichletWeight;
  }

  public int getNumberOfAllAvailableMoves() {
    return numberOfAllAvailableMoves;
  }

  public void setNumberOfAllAvailableMoves(int numberOfAllAvailableMoves) {
    this.numberOfAllAvailableMoves = numberOfAllAvailableMoves;
  }

  public int getNumberOfEpisodesBeforePotentialUpdate() {
    return numberOfEpisodesBeforePotentialUpdate;
  }

  public void setNumberOfEpisodesBeforePotentialUpdate(int numberOfEpisodesBeforePotentialUpdate) {
    this.numberOfEpisodesBeforePotentialUpdate = numberOfEpisodesBeforePotentialUpdate;
  }

  public int getNumberOfEpisodeThreads() {
    return numberOfEpisodeThreads;
  }

  public void setNumberOfEpisodeThreads(int numberOfEpisodeThreads) {
    this.numberOfEpisodeThreads = numberOfEpisodeThreads;
  }

  public boolean isContinueTraining() {
    return continueTraining;
  }

  public void setContinueTraining(boolean continueTraining) {
    this.continueTraining = continueTraining;
  }

  public int getInitialIteration() {
    return this.initialIteration;
  }

  public void setInitialIteration(int initialIteration) {
    this.initialIteration = initialIteration;
  }

  public int getNumberOfIterations() {
    return numberOfIterations;
  }

  public void setNumberOfIterations(int numberOfIterations) {
    this.numberOfIterations = numberOfIterations;
  }

  public int getCheckPointIterationsFrequency() {
    return checkPointIterationsFrequency;
  }

  public void setCheckPointIterationsFrequency(int checkPointIterationsFrequency) {
    this.checkPointIterationsFrequency = checkPointIterationsFrequency;
  }

  
  public double getCurrentTemperature(int iteration, int moveNumber) {

    if (getFromNumberOfIterationsTemperatureZero() >= 0 && iteration >= getFromNumberOfIterationsTemperatureZero() ||
        getFromNumberOfMovesTemperatureZero() >= 0 && moveNumber >= getFromNumberOfMovesTemperatureZero()) {
      return reducedTemperature;
    }
    
    return AdversaryLearningConstants.ONE;
  }
 
  public int getFromNumberOfIterationsTemperatureZero() {
    return fromNumberOfIterationsReducedTemperature;
  }

  public void setFromNumberOfIterationsTemperatureZero(int fromNumberOfIterationsTemperatureZero) {
    this.fromNumberOfIterationsReducedTemperature = fromNumberOfIterationsTemperatureZero;
  }
  
  public int getFromNumberOfMovesTemperatureZero() {
    return fromNumberOfMovesReducedTemperature;
  }

  public void setFromNumberOfMovesTemperatureZero(int fromNumberOfMovesTemperatureZero) {
    this.fromNumberOfMovesReducedTemperature = fromNumberOfMovesTemperatureZero;
  }

  public double getReducedTemperature() {
    return reducedTemperature;
  }
  
  public void setReducedTemperature(double reducedTemperature) {
    this.reducedTemperature = reducedTemperature;
  }
  
  public int getCurrentMaxTrainExamplesHistory(int currentIteration) {
    
    if (currentIteration >= getMaxTrainExamplesHistoryFromIteration()) {
      
      return getMaxTrainExamplesHistory();
    }
    
    return (int) (currentIteration / ((float) getMaxTrainExamplesHistoryFromIteration()) * getMaxTrainExamplesHistory());
  }
  
  public int getMaxTrainExamplesHistory() {
    
    return maxTrainExamplesHistory;
  }

  public void setMaxTrainExamplesHistory(int maxTrainExamplesHistory) {

    this.maxTrainExamplesHistory = maxTrainExamplesHistory;
  }

  public int getMaxTrainExamplesHistoryFromIteration() {
    
    return maxTrainExamplesHistoryFromIteration;
  }

  public void setMaxTrainExamplesHistoryFromIteration(int maxTrainExamplesHistoryFromIteration) {

    this.maxTrainExamplesHistoryFromIteration = maxTrainExamplesHistoryFromIteration;
  }

  public double getuctConstantFactor() {
    return uctConstantFactor;
  }

  public void setUctConstantFactor(double uctConstantFactor) {
    this.uctConstantFactor = uctConstantFactor;
  }

  public int getNumberOfMonteCarloSimulations() {
    return numberOfMonteCarloSimulations;
  }

  public void setNumberOfMonteCarloSimulations(int nummberOfMonteCarloSimulations) {
    this.numberOfMonteCarloSimulations = nummberOfMonteCarloSimulations;
  }
  
  public static String getAbsolutePathFrom(String fileName) {
    
    Path filePath = Paths.get(fileName);
    
    if (!filePath.isAbsolute()) {

      Path currentPath = Paths.get(StringUtils.EMPTY).toAbsolutePath();
      return String.valueOf(Paths.get(String.valueOf(currentPath), fileName));
    }
    
    return String.valueOf(filePath);
  }

  public String getBestModelFileName() {
    return bestModelFileName;
  }

  public void setBestModelFileName(String bestModelFileName) {
    this.bestModelFileName = bestModelFileName;
  }

  public String getTrainExamplesFileName() {
    return trainExamplesFileName;
  }

  public void setTrainExamplesFileName(String trainExamplesFileName) {
    this.trainExamplesFileName = trainExamplesFileName;
  }
}
