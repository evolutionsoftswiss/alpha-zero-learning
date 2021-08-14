package ch.evolutionsoft.rl;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.schedule.ISchedule;

import ch.evolutionsoft.net.game.NeuralNetConstants;

/**
 * {@link AdversaryLearningConfiguration} defines several configuration parameters
 * affecting the behavior of alpha zero learning.
 * 
 * @author evolutionsoft
 */
public class AdversaryLearningConfiguration {

  /**
   * Fixed {@link ComputationGraph} learning rate, currently the save option between
   * restored tempmodel.bin or bestmodel.bin from files. See also comment for
   * learningRateSchedule.
   */
  double learningRate;
  
  /**
   * A learningRateSchedule defining different learning rates in function of the
   * number of performed net iterations meaning calls to {@link ComputationGraph} fit method here.
   * 
   * Be careful when {@link ComputationGraph} gets restored from files tempmodel.bin or bestmodel.bin.
   * Double.NaN values were observed after restoring with learningRateSchedule.
   * 
   * The iterations deciding the learning rate for ISchedule is not directly related to alpha zero 
   * numberOfIterations. With alwaysUpdateNeuralNetwork = true it is also dependent of 
   * numberOfEpisodesBeforePotentialUpdate. For the default TicTacToe {@link AdversaryLearningConfiguration} 
   * with numberOfEpisodesBeforePotentialUpdate = 5. The relation between ISchedule iterations and Alpha Zero
   * numberOfIterations is then: 5 * learningRateIterations = numberOfIterations.
   * 
   * If alwaysUpdateNeuralNetwork = false the learningRateSchedule iterations depend on the number of 
   * performed {@link ComputationGraph} updates using the fit method.
   * 
   */
  ISchedule learningRateSchedule;

  int batchSize;

  /**
   * Value of the dirichlet alpha used to add noise to move probability distributions.
   * TicTacToe uses a rather high default value compared to other games known values from Alpha Zero.
   */
  double dirichletAlpha;

  /**
   * Weight of the dirichlet noise added to currently known move probabilities.
   * The higher default TicTacToe value helped to generate more unique training examples.
   * The known unique 4520 training examples (from dl4j supervised leanring project) is not
   * completely generated after the current numberOfIterations.
   */
  double dirichletWeight;

  /**
   * True means Alpha Zero approach to update the neural net without games comparing the
   * win rate of different neural net versions. After numberOfEpisodesBeforePotentialUpdate
   * the Alpha Zero net gets always updated. gamesToGetNewNetworkWinRatio and 
   * updateGamesNewNetworkWinRatioThreshold are irrelevant with this configuration set to true.
   * 
   * False means the AlphaGo Zero approach by running games with different neural net versions.
   * gamesToGetNewNetworkWinRatio and updateGamesNewNetworkWinRatioThreshold are used to
   * decide if the neural net gets updated or not.
   */
  boolean alwaysUpdateNeuralNetwork;

  /**
   * Number of total games to perform before deciding to update neural net or not.
   * Only relevant with alwaysUpdateNeuralNetwork = false.
   */
  int gamesToGetNewNetworkWinRatio;

  /**
   * Win ratio minimum to perform an update of the neural network.
   * 
   *  updateAfterBetterPlayout = 
   *    (newNeuralNetVersionWins + 0.5 * draws) /
   *    (double) (newNeuralNetVersionWins + oldNeuralNetVersionWins + 0.5 * draws) > updateGamesNewNetworkWinRatioThreshold;
   */
  double updateGamesNewNetworkWinRatioThreshold;

  /**
   * An Alpha Zero episode is one game from start to end. Each episode generates potentially new
   * training examples used to train the neural net. numberOfEpisodes defines how much times a game 
   * will be run from start to end to gather training examples before a potential neural net update.
   */
  int numberOfEpisodes;

  /**
   * Mainly used to continue training after program termination.
   * Only iterationStart > 1 causes a restore of saved tempmodel.bin, bestmodel.bin and trainexamples.obj.
   * If you decide to run additional 1000 iterations after 4000 performed iterations with
   * saved latest values after program termination,
   * you can use iterationStart = 4001 and numberOfIterations = 1000.
   */
  int iterationStart;

  /**
   * numberOfIterations here means the total number of Alpha Zero iterations.
   */
  int numberOfIterations;

  /**
   * After checkPointIterationsFrequency store additional files containing the current
   * model and training examples.
   */
  int checkPointIterationsFrequency;

  /**
   * When the temperature used in {@link MonteCarloSearch} getActionValues() should become 0.
   * Currently only 1 or 0 are used. Too small values > 0 can cause overflows.
   * A temperature == 0 will lead to move action probabilities all zero, expect
   * one being one. Temperatures > 0 keep probabilities > 0 for all move actions
   * in function of the number of visits during {@link MonteCarloSearch}.
   */
  int fromNumberOfIterationsTemperatureZero;
  
  /**
   * Currently unused in TicTacToe example implementation.
   * Also in early iterations use zero temperature after having reached the
   * specified number of moves in an alpha zero iteration.
   */
  int fromNumberOfMovesTemperatureZero;

  /**
   * The maximum number of train examples to keep in history and reuse for neural net fit.
   * TicTacToe never exceeds the used value of 5000.
   * Typical values for Go 19x19 are 1 or 2 million.
   */
  int maxTrainExamplesHistory;

  /**
   * {@link MonteCarloSearch} parameter influencing exploration / exploitation of
   * different move actions. Currently 1.5 is used.
   */
  double uctConstantFactor;

  /**
   * How much single playout steps should {@link MonteCarloSearch} perform.
   * TicTacToe example implementation uses 30.
   * Typical values for Go 9x9 and Go 19x19 would be 400 and 1600.
   */
  int numberOfMonteCarloSimulations;

  /**
   * Default initial values for TicTacToe example implementation.
   * 
   * @author evolutionsoft
   */
  public static class Builder {

    double learningRate = 1e-4;
    ISchedule learningRateSchedule;
    int batchSize = 8192;

    double dirichletAlpha = 1.1;
    double dirichletWeight = 0.45;
    boolean alwaysUpdateNeuralNetwork = true;
    int gamesToGetNewNetworkWinRatio = 36;
    double updateGamesNewNetworkWinRatioThreshold = 0.55;
    int numberEpisodes = 10;
    int iterationStart = 1;
    int numberOfIterations = 250;
    int checkPointIterationsFrequency = 50;
    int fromNumberOfIterationsTemperatureZero = -1;
    int fromNumberOfMovesTemperatureZero = 3;
    int maxTrainExamplesHistory = 5000;

    double uctConstantFactor = 0.8;
    int numberOfMonteCarloSimulations = 30;
    
    public AdversaryLearningConfiguration build() {
      
      AdversaryLearningConfiguration configuration = new AdversaryLearningConfiguration();
      
      configuration.learningRate = learningRate;
      configuration.learningRateSchedule = learningRateSchedule;
      configuration.batchSize = batchSize;
      configuration.dirichletAlpha = dirichletAlpha;
      configuration.dirichletWeight = dirichletWeight;
      configuration.alwaysUpdateNeuralNetwork = alwaysUpdateNeuralNetwork;
      configuration.gamesToGetNewNetworkWinRatio = gamesToGetNewNetworkWinRatio;
      configuration.updateGamesNewNetworkWinRatioThreshold = updateGamesNewNetworkWinRatioThreshold;
      configuration.numberOfEpisodes = numberEpisodes;
      configuration.iterationStart = iterationStart;
      configuration.numberOfIterations = numberOfIterations;
      configuration.checkPointIterationsFrequency = checkPointIterationsFrequency;
      configuration.fromNumberOfIterationsTemperatureZero = fromNumberOfIterationsTemperatureZero;
      configuration.fromNumberOfMovesTemperatureZero = fromNumberOfMovesTemperatureZero;
      configuration.maxTrainExamplesHistory = maxTrainExamplesHistory;
      configuration.uctConstantFactor = uctConstantFactor;
      configuration.numberOfMonteCarloSimulations = numberOfMonteCarloSimulations;
      
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
      this.gamesToGetNewNetworkWinRatio = numberOfGamesToDecideUpdate;
      return this;
    }

    public Builder updateNeuralNetworkThreshold(double updateNeuralNetworkThreshold) {
      this.updateGamesNewNetworkWinRatioThreshold = updateNeuralNetworkThreshold;
      return this;
    }

    public Builder numberOfIterationsBeforePotentialUpdate(int numberOfEpisodesBeforePotentialUpdate) {
      this.numberEpisodes = numberOfEpisodesBeforePotentialUpdate;
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
  }
  
  public String toString() {
    
    return " learningRate: " + (null != this.learningRateSchedule ? this.learningRateSchedule : this.learningRate) +
        "\n batch size: " + this.batchSize +
        "\n dirichletAlpha: " + this.dirichletAlpha + 
        "\n dirichletWeight: " + this.dirichletWeight +
        "\n alwaysUpdateNeuralNetwork: " + this.alwaysUpdateNeuralNetwork +
        "\n gamesToGetNewNetworkWinRatio: " + (this.alwaysUpdateNeuralNetwork ? "-" : this.gamesToGetNewNetworkWinRatio) +
        "\n updateGamesNewNetworkWinRatioThreshold: " + (this.alwaysUpdateNeuralNetwork ? "-" : this.updateGamesNewNetworkWinRatioThreshold) +
        "\n numberOfEpisodesBeforePotentialUpdate: " + this.numberOfEpisodes + 
        "\n iterationStart: " + this.iterationStart + 
        "\n numberOfIterations: " + this.numberOfIterations +
        "\n checkPointIterationsFrequency: " + this.checkPointIterationsFrequency +
        "\n fromNumberOfIterationsTemperatureZero: " + this.fromNumberOfIterationsTemperatureZero +
        "\n fromNumberOfMovesTemperatureZero: " + this.fromNumberOfMovesTemperatureZero +
        "\n maxTrainExamplesHistory: " + this.maxTrainExamplesHistory +
        "\n cpUct: " + this.uctConstantFactor +
        "\n numberOfMonteCarloSimulations: " + this.numberOfMonteCarloSimulations;
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

  public int getGamesToGetNewNetworkWinRatio() {
    return gamesToGetNewNetworkWinRatio;
  }

  public void setGamesToGetNewNetworkWinRatio(int numberOfGamesToDecideUpdate) {
    this.gamesToGetNewNetworkWinRatio = numberOfGamesToDecideUpdate;
  }

  public double getUpdateGamesNewNetworkWinRatioThreshold() {
    return updateGamesNewNetworkWinRatioThreshold;
  }

  public void setUpdateGamesNewNetworkWinRatioThreshold(double updateNeuralNetworkThreshold) {
    this.updateGamesNewNetworkWinRatioThreshold = updateNeuralNetworkThreshold;
  }

  public int getNumberOfEpisodes() {
    return numberOfEpisodes;
  }

  public void setNumberOfEpisodes(int numberOfEpisodesBeforePotentialUpdate) {
    this.numberOfEpisodes = numberOfEpisodesBeforePotentialUpdate;
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
    
    return NeuralNetConstants.ONE;
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
}
