package ch.evolutionsoft.rl;

import org.nd4j.linalg.schedule.ISchedule;

import ch.evolutionsoft.net.game.NeuralNetConstants;

/**
 * 
 * 
 * @author evolutionsoft
 */
public class AdversaryLearningConfiguration {
  
  double learningRate;
  ISchedule learningRateSchedule;

  double dirichletAlpha;

  double dirichletWeight;
  boolean alwaysUpdateNeuralNetwork;
  int gamesToGetNewNetworkWinRatio;
  double updateGamesNewNetworkWinRatioThreshold;
  int numberOfEpisodesBeforePotentialUpdate;
  int iterationStart;
  int numberOfIterations;
  int checkPointIterationsFrequency;
  int fromNumberOfIterationsTemperatureZero;
  int fromNumberOfMovesTemperatureZero;
  int maxTrainExamplesHistory;

  double cpUct;
  int nummberOfMonteCarloSimulations;

  public static class Builder {

    double learningRate = 1e-4;
    ISchedule learningRateSchedule;

    double dirichletAlpha = 1.1;
    double dirichletWeight = 0.45;
    boolean alwaysUpdateNeuralNetwork = true;
    int gamesToGetNewNetworkWinRatio = 36;
    double updateGamesNewNetworkWinRatioThreshold = 0.55;
    int numberOfEpisodesBeforePotentialUpdate = 5;
    int iterationStart = 1;
    int numberOfIterations = 4000;
    int checkPointIterationsFrequency = 1000;
    int fromNumberOfIterationsTemperatureZero = 2500;
    int fromNumberOfMovesTemperatureZero = -1;
    int maxTrainExamplesHistory = 5000;

    double cpUct = 1.0;
    int numberOfMonteCarloSimulations = 50;
    
    public AdversaryLearningConfiguration build() {
      
      AdversaryLearningConfiguration configuration = new AdversaryLearningConfiguration();
      
      configuration.learningRate = learningRate;
      configuration.learningRateSchedule = learningRateSchedule;
      configuration.dirichletAlpha = dirichletAlpha;
      configuration.dirichletWeight = dirichletWeight;
      configuration.alwaysUpdateNeuralNetwork = alwaysUpdateNeuralNetwork;
      configuration.gamesToGetNewNetworkWinRatio = gamesToGetNewNetworkWinRatio;
      configuration.updateGamesNewNetworkWinRatioThreshold = updateGamesNewNetworkWinRatioThreshold;
      configuration.numberOfEpisodesBeforePotentialUpdate = numberOfEpisodesBeforePotentialUpdate;
      configuration.iterationStart = iterationStart;
      configuration.numberOfIterations = numberOfIterations;
      configuration.checkPointIterationsFrequency = checkPointIterationsFrequency;
      configuration.fromNumberOfIterationsTemperatureZero = fromNumberOfIterationsTemperatureZero;
      configuration.fromNumberOfMovesTemperatureZero = fromNumberOfMovesTemperatureZero;
      configuration.maxTrainExamplesHistory = maxTrainExamplesHistory;
      configuration.cpUct = cpUct;
      configuration.nummberOfMonteCarloSimulations = numberOfMonteCarloSimulations;
      
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
    
    public ISchedule getLearningRateSchedule() {
      return learningRateSchedule;
    }

    public void setLearningRateSchedule(ISchedule learningRateSchedule) {
      this.learningRateSchedule = learningRateSchedule;
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

    public Builder numberOfEpisodesBeforePotentialUpdate(int numberOfEpisodesBeforePotentialUpdate) {
      this.numberOfEpisodesBeforePotentialUpdate = numberOfEpisodesBeforePotentialUpdate;
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

    public Builder cpUct(double cpUct) {
      this.cpUct = cpUct;
      return this;
    }

    public Builder numberOfMonteCarloSimulations(int numberOfMonteCarloSimulations) {
      this.numberOfMonteCarloSimulations = numberOfMonteCarloSimulations;
      return this;
    }
  }
  
  public String toString() {
    
    return " learningRate: " + (null == this.learningRateSchedule ? "-" : this.learningRate) +
        "\n dirichletAlpha: " + this.dirichletAlpha + 
        "\n dirichletWeight: " + this.dirichletWeight +
        "\n alwaysUpdateNeuralNetwork: " + this.alwaysUpdateNeuralNetwork +
        "\n gamesToGetNewNetworkWinRatio: " + (this.alwaysUpdateNeuralNetwork ? "-" : this.gamesToGetNewNetworkWinRatio) +
        "\n updateGamesNewNetworkWinRatioThreshold: " + (this.alwaysUpdateNeuralNetwork ? "-" : this.updateGamesNewNetworkWinRatioThreshold) +
        "\n numberOfEpisodesBeforePotentialUpdate: " + this.numberOfEpisodesBeforePotentialUpdate + 
        "\n iterationStart: " + this.iterationStart + 
        "\n numberOfIterations: " + this.numberOfIterations +
        "\n fromNumberOfIterationsTemperatureZero: " + this.fromNumberOfIterationsTemperatureZero +
        "\n fromNumberOfMovesTemperatureZero: " + this.fromNumberOfMovesTemperatureZero +
        "\n maxTrainExamplesHistory: " + this.maxTrainExamplesHistory +
        "\n cpUct: " + this.cpUct +
        "\n numberOfMonteCarloSimulations: " + this.nummberOfMonteCarloSimulations;
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

  public int getNumberOfEpisodesBeforePotentialUpdate() {
    return numberOfEpisodesBeforePotentialUpdate;
  }

  public void setNumberOfEpisodesBeforePotentialUpdate(int numberOfEpisodesBeforePotentialUpdate) {
    this.numberOfEpisodesBeforePotentialUpdate = numberOfEpisodesBeforePotentialUpdate;
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
    
    if (iteration >= getFromNumberOfIterationsTemperatureZero() ||
        moveNumber > getFromNumberOfMovesTemperatureZero()) {
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

  public double getCpUct() {
    return cpUct;
  }

  public void setCpUct(double cpUct) {
    this.cpUct = cpUct;
  }

  public int getNummberOfMonteCarloSimulations() {
    return nummberOfMonteCarloSimulations;
  }

  public void setNummberOfMonteCarloSimulations(int nummberOfMonteCarloSimulations) {
    this.nummberOfMonteCarloSimulations = nummberOfMonteCarloSimulations;
  }
}
