package ch.evolutionsoft.rl;

import java.io.File;
import java.nio.file.Paths;

import org.apache.commons.lang3.StringUtils;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.schedule.ISchedule;

/**
 * {@link AdversaryLearningConfiguration} defines several configuration parameters
 * affecting the behavior of alpha zero learning.
 * 
 * @author evolutionsoft
 */
public class AdversaryLearningConfiguration {

  /**
   * Fixed {@link ComputationGraph} learning rate
   */
  private double learningRate;
  
  /**
   * A learningRateSchedule defining different learning rates in function of the
   * number of performed net iterations meaning calls to {@link ComputationGraph} fit method here.
   * 
   * The iterations deciding the learning rate for ISchedule is only directly related to alpha zero 
   * numberOfIterations with alwaysUpdateNeuralNetwork = true.
   * Otherwise with alwaysUpdateNeuralNetwork = false it is also dependent of 
   * performed {@link ComputationGraph} updates using the fit method.
   */
  private ISchedule learningRateSchedule;

  /**
   * Size of mini batches used to perform {@link ComputationGraph} updates with fit.
   *
   * TicTacToe currently uses a value greater than all possible {@link AdversaryTrainingExample}
   * leading to use one single batch always.
   */
  private int batchSize;

  /**
   * Value of the dirichlet alpha used to add noise to move probability distributions.
   * TicTacToe uses a rather high default value compared to other games known values from Alpha Zero.
   */
  private double dirichletAlpha;

  /**
   * Weight of the dirichlet noise added to currently known move probabilities.
   * The higher default TicTacToe value helped to generate more unique training examples.
   * The known unique 4520 training examples (from dl4j supervised learning project) is not
   * completely generated after the current numberOfIterations.
   */
  private double dirichletWeight;

  /**
   * True means Alpha Zero approach to update the neural net without games comparing the
   * win rate of different neural net versions. After numberOfEpisodesBeforePotentialUpdate
   * the Alpha Zero net gets always updated. gamesToGetNewNetworkWinRatio and 
   * updateGamesNewNetworkWinRatioThreshold are irrelevant with this configuration set to true.
   * 
   * False uses the AlphaGo Zero approach by running games with different neural net versions.
   * gamesToGetNewNetworkWinRatio and updateGamesNewNetworkWinRatioThreshold are used to
   * decide if the neural net gets updated or not.
   */
  private boolean alwaysUpdateNeuralNetwork;

  /**
   * Number of total games to perform before deciding to update neural net or not.
   * Only relevant with alwaysUpdateNeuralNetwork = false.
   */
  private int numberOfGamesToDecideUpdate;

  /**
   * Win ratio minimum to perform an update of the neural network.
   * 
   *  updateAfterBetterPlayout = 
   *    (newNeuralNetVersionWins + 0.5 * draws) /
   *    (double) (newNeuralNetVersionWins + oldNeuralNetVersionWins + 0.5 * draws) > gamesWinRatioThresholdNewNetworkUpdate;
   */
  private double gamesWinRatioThresholdNewNetworkUpdate;

  /**
   * numberOfIterationsBeforePotentialUpdate stands for numberOfEpisodes.
   * An Alpha Zero episode is one game from start to end. Each episode generates potentially new
   * training examples used to train the neural net. numberOfIterationsBeforePotentialUpdate defines how much times a game 
   * will be run from start to end to gather training examples before a potential neural net update.
   */
  private int numberOfIterationsBeforePotentialUpdate;

  /**
   * Mainly used to continue training after program termination.
   * Only iterationStart > 1 causes a restore of saved tempmodel.bin, bestmodel.bin and trainexamples.obj.
   * If you decide to run additional 1000 iterations after 4000 performed iterations with
   * saved latest values after program termination,
   * you can use iterationStart = 4001 and numberOfIterations = 1000.
   */
  private int iterationStart;

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
   * Currently only 1 or 0 are used. Too small values > 0 can cause overflows.
   * A temperature == 0 will lead to move action probabilities all zero, expect
   * one being one. Temperatures > 0 keep probabilities > 0 for all move actions
   * in function of the number of visits during {@link MonteCarloTreeSearch}.
   */
  private int fromNumberOfIterationsTemperatureZero;
  
  /**
   * Currently used approach in TicTacToe example implementation.
   * Also in early iterations use zero temperature after having reached the
   * specified number of moves in an alpha zero iteration.
   */
  private int fromNumberOfMovesTemperatureZero;

  /**
   * The maximum number of train examples to keep in history and reuse for neural net fit.
   * TicTacToe never exceeds the used value of 5000.
   * Typical values for Go 19x19 are 1 or 2 million.
   */
  private int maxTrainExamplesHistory;

  /**
   * {@link MonteCarloTreeSearch} parameter influencing exploration / exploitation of
   * different move actions. TicTacToe uses 0.8.
   */
  private double uctConstantFactor;

  /**
   * How much single playout steps should {@link MonteCarloTreeSearch} perform.
   * TicTacToe example implementation uses 30.
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

    private double learningRate = 1e-4;
    private ISchedule learningRateSchedule;
    private int batchSize = 8192;

    private double dirichletAlpha = 1.1;
    private double dirichletWeight = 0.45;
    private boolean alwaysUpdateNeuralNetwork = true;
    private int numberOfGamesToDecideUpdate = 36;
    private double gamesWinRatioThresholdNewNetworkUpdate = 0.55;
    private int numberOfIterationsBeforePotentialUpdate = 10;
    private int iterationStart = 1;
    private int numberOfIterations = 250;
    private int checkPointIterationsFrequency = 50;
    private int fromNumberOfIterationsTemperatureZero = -1;
    private int fromNumberOfMovesTemperatureZero = 3;
    private int maxTrainExamplesHistory = 5000;

    private String bestModelFileName = "bestmodel.bin";
    private String trainExamplesFileName = "trainExamples.obj";

    private double uctConstantFactor = 0.8;
    private int numberOfMonteCarloSimulations = 30;
    
    public AdversaryLearningConfiguration build() {
      
      AdversaryLearningConfiguration configuration = new AdversaryLearningConfiguration();
      
      configuration.learningRate = learningRate;
      configuration.learningRateSchedule = learningRateSchedule;
      configuration.batchSize = batchSize;
      configuration.dirichletAlpha = dirichletAlpha;
      configuration.dirichletWeight = dirichletWeight;
      configuration.alwaysUpdateNeuralNetwork = alwaysUpdateNeuralNetwork;
      configuration.numberOfGamesToDecideUpdate = numberOfGamesToDecideUpdate;
      configuration.gamesWinRatioThresholdNewNetworkUpdate = gamesWinRatioThresholdNewNetworkUpdate;
      configuration.numberOfIterationsBeforePotentialUpdate = numberOfIterationsBeforePotentialUpdate;
      configuration.iterationStart = iterationStart;
      configuration.numberOfIterations = numberOfIterations;
      configuration.checkPointIterationsFrequency = checkPointIterationsFrequency;
      configuration.fromNumberOfIterationsTemperatureZero = fromNumberOfIterationsTemperatureZero;
      configuration.fromNumberOfMovesTemperatureZero = fromNumberOfMovesTemperatureZero;
      configuration.maxTrainExamplesHistory = maxTrainExamplesHistory;
      configuration.uctConstantFactor = uctConstantFactor;
      configuration.numberOfMonteCarloSimulations = numberOfMonteCarloSimulations;
      configuration.bestModelFileName = bestModelFileName;
      configuration.trainExamplesFileName = trainExamplesFileName;
      
      return configuration;
    }
 
    public Builder learningRate(double neuralNetworkLearningRate) {
      this.learningRate = neuralNetworkLearningRate;
      return this;
    }

    public Builder learningRateSchedule(ISchedule learningRateSchedule) {
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

    public Builder fromNumberOfIterationsTemperatureZero(int fromNumberOfIterationsTemperatureZero) {
      this.fromNumberOfIterationsTemperatureZero = fromNumberOfIterationsTemperatureZero;
      return this;
    }

    public Builder fromNumberOfMovesTemperatureZero(int fromNumberOfMovesTemperatureZero) {
      this.fromNumberOfMovesTemperatureZero = fromNumberOfMovesTemperatureZero;
      return this;
    }
    
    public Builder alwaysUpdateNeuralNetwork(boolean alwaysUpdateNeuralNetwork) {
      this.alwaysUpdateNeuralNetwork = alwaysUpdateNeuralNetwork;
      return this;
    }
    
    public Builder numberOfGamesToDecideUpdate(int numberOfGamesToDecideUpdate) {
      this.numberOfGamesToDecideUpdate = numberOfGamesToDecideUpdate;
      return this;
    }

    public Builder gamesWinRatioThresholdNewNetworkUpdate(double gamesWinRatioThresholdNewNetworkUpdate) {
      this.gamesWinRatioThresholdNewNetworkUpdate = gamesWinRatioThresholdNewNetworkUpdate;
      return this;
    }

    public Builder numberOfIterationsBeforePotentialUpdate(int numberOfEpisodesBeforePotentialUpdate) {
      this.numberOfIterationsBeforePotentialUpdate = numberOfEpisodesBeforePotentialUpdate;
      return this;
    }
    
    public Builder iterationStart(int iterationStart) {
      this.iterationStart = iterationStart;
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
    
    return " learningRate: " + (null != this.learningRateSchedule ? this.learningRateSchedule : this.learningRate) +
        "\n batch size: " + this.batchSize +
        "\n dirichletAlpha: " + this.dirichletAlpha + 
        "\n dirichletWeight: " + this.dirichletWeight +
        "\n alwaysUpdateNeuralNetwork: " + this.alwaysUpdateNeuralNetwork +
        "\n gamesToGetNewNetworkWinRatio: " + (this.alwaysUpdateNeuralNetwork ? "-" : this.numberOfGamesToDecideUpdate) +
        "\n gamesWinRatioThresholdNewNetworkUpdate: " + (this.alwaysUpdateNeuralNetwork ? "-" : this.gamesWinRatioThresholdNewNetworkUpdate) +
        "\n numberOfEpisodesBeforePotentialUpdate: " + this.numberOfIterationsBeforePotentialUpdate + 
        "\n iterationStart: " + this.iterationStart + 
        "\n numberOfIterations: " + this.numberOfIterations +
        "\n checkPointIterationsFrequency: " + this.checkPointIterationsFrequency +
        "\n fromNumberOfIterationsTemperatureZero: " + this.fromNumberOfIterationsTemperatureZero +
        "\n fromNumberOfMovesTemperatureZero: " + this.fromNumberOfMovesTemperatureZero +
        "\n maxTrainExamplesHistory: " + this.maxTrainExamplesHistory +
        "\n cpUct: " + this.uctConstantFactor +
        "\n numberOfMonteCarloSimulations: " + this.numberOfMonteCarloSimulations +
        "\n bestModelFileName: " + getAbsoluteModelPathFrom(this.bestModelFileName) +
        "\n trainExamplesFileName: " + getAbsoluteModelPathFrom(this.trainExamplesFileName);
  }

  public double getLearningRate() {
    return learningRate;
  }

  public void setLearningRate(double neuralNetworkLearningRate) {
    this.learningRate = neuralNetworkLearningRate;
  }
  
  public ISchedule getLearningRateSchedule() {
    return learningRateSchedule;
  }

  public void setLearningRateSchedule(ISchedule learningRateSchedule) {
    this.learningRateSchedule = learningRateSchedule;
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

  public boolean isAlwaysUpdateNeuralNetwork() {
    return alwaysUpdateNeuralNetwork;
  }

  public void setAlwaysUpdateNeuralNetwork(boolean alwaysUpdateNeuralNetwork) {
    this.alwaysUpdateNeuralNetwork = alwaysUpdateNeuralNetwork;
  }

  public int getNumberOfGamesToDecideUpdate() {
    return numberOfGamesToDecideUpdate;
  }

  public void setNumberOfGamesToDecideUpdate(int numberOfGamesToDecideUpdate) {
    this.numberOfGamesToDecideUpdate = numberOfGamesToDecideUpdate;
  }

  public double getGamesWinRatioThresholdNewNetworkUpdate() {
    return gamesWinRatioThresholdNewNetworkUpdate;
  }

  public void setGamesWinRatioThresholdNewNetworkUpdate(double gamesWinRatioThresholdNewNetworkUpdate) {
    this.gamesWinRatioThresholdNewNetworkUpdate = gamesWinRatioThresholdNewNetworkUpdate;
  }

  public int getNumberOfIterationsBeforePotentialUpdate() {
    return numberOfIterationsBeforePotentialUpdate;
  }

  public void setNumberOfIterationsBeforePotentialUpdate(int numberOfEpisodesBeforePotentialUpdate) {
    this.numberOfIterationsBeforePotentialUpdate = numberOfEpisodesBeforePotentialUpdate;
  }
  
  public int getIterationStart() {
    return this.iterationStart;
  }
  
  public void setIterationStart(int iterationStart) {
    this.iterationStart = iterationStart;
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
      return 0;
    }
    
    return AdversaryLearningConstants.ONE;
  }
 
  public int getFromNumberOfIterationsTemperatureZero() {
    return fromNumberOfIterationsTemperatureZero;
  }

  public void setFromNumberOfIterationsTemperatureZero(int fromNumberOfIterationsTemperatureZero) {
    this.fromNumberOfIterationsTemperatureZero = fromNumberOfIterationsTemperatureZero;
  }
  
  public int getFromNumberOfMovesTemperatureZero() {
    return fromNumberOfMovesTemperatureZero;
  }

  public void setFromNumberOfMovesTemperatureZero(int fromNumberOfMovesTemperatureZero) {
    this.fromNumberOfMovesTemperatureZero = fromNumberOfMovesTemperatureZero;
  }

  public int getMaxTrainExamplesHistory() {
    return maxTrainExamplesHistory;
  }

  public void setMaxTrainExamplesHistory(int maxTrainExamplesHistory) {
    this.maxTrainExamplesHistory = maxTrainExamplesHistory;
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
  
  public String getAbsoluteModelPathFrom(String modelName) {
  
    String currentPath = String.valueOf(Paths.get(StringUtils.EMPTY).toAbsolutePath());
    
    return currentPath + File.separator + modelName;
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
