package ch.evolutionsoft.rl;

public class AdversaryLearningConfiguration {
  
  double neuralNetworkLearningRate;
  
  double dirichletAlpha;

  double dirichletWeight;
  boolean alwaysUpdateNeuralNetwork;
  int gamesToGetNewNetworkWinRatio;
  double updateGamesNewNetworkWinRatioThreshold;
  int numberOfEpisodesBeforePotentialUpdate;
  int iterationStart;
  int numberOfIterations;
  int maxTrainExamplesHistory;

  double cpUct;
  int nummberOfMonteCarloSimulations;

  public static class Builder {

    double neuralNetworkLearningRate = 1e-3;
    
    double dirichletAlpha = 1.5;
    double dirichletWeight = 0.55;
    boolean alwaysUpdateNeuralNetwork = true;
    int gamesToGetNewNetworkWinRatio = 36;
    double updateGamesNewNetworkWinRatioThreshold = 0.55;
    int numberOfEpisodesBeforePotentialUpdate = 20;
    int iterationStart = 1;
    int numberOfIterations = 2500;
    int maxTrainExamplesHistory = 5000;

    double cpUct = 1.0;
    int numberOfMonteCarloSimulations = 25;
    
    public AdversaryLearningConfiguration build() {
      
      AdversaryLearningConfiguration configuration = new AdversaryLearningConfiguration();
      
      configuration.neuralNetworkLearningRate = neuralNetworkLearningRate;
      configuration.dirichletAlpha = dirichletAlpha;
      configuration.dirichletWeight = dirichletWeight;
      configuration.alwaysUpdateNeuralNetwork = alwaysUpdateNeuralNetwork;
      configuration.gamesToGetNewNetworkWinRatio = gamesToGetNewNetworkWinRatio;
      configuration.updateGamesNewNetworkWinRatioThreshold = updateGamesNewNetworkWinRatioThreshold;
      configuration.numberOfEpisodesBeforePotentialUpdate = numberOfEpisodesBeforePotentialUpdate;
      configuration.iterationStart = iterationStart;
      configuration.numberOfIterations = numberOfIterations;
      configuration.maxTrainExamplesHistory = maxTrainExamplesHistory;
      configuration.cpUct = cpUct;
      configuration.nummberOfMonteCarloSimulations = numberOfMonteCarloSimulations;
      
      return configuration;
    }
 
    public Builder neuralNetworkLearningRate(double neuralNetworkLearningRate) {
      this.neuralNetworkLearningRate = neuralNetworkLearningRate;
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
    
    return " neuralNetworkLearningRate: " + this.neuralNetworkLearningRate + 
        "\n dirichletAlpha: " + this.dirichletAlpha + 
        "\n dirichletWeight: " + this.dirichletWeight +
        "\n alwaysUpdateNeuralNetwork: " + this.alwaysUpdateNeuralNetwork +
        "\n gamesToGetNewNetworkWinRatio: " + (this.alwaysUpdateNeuralNetwork ? "-" : this.gamesToGetNewNetworkWinRatio) +
        "\n updateGamesNewNetworkWinRatioThreshold: " + (this.alwaysUpdateNeuralNetwork ? "-" : this.updateGamesNewNetworkWinRatioThreshold) +
        "\n numberOfEpisodesBeforePotentialUpdate: " + this.numberOfEpisodesBeforePotentialUpdate + 
        "\n iterationStart: " + this.iterationStart + 
        "\n numberOfIterations: " + this.numberOfIterations +
        "\n maxTrainExamplesHistory: " + this.maxTrainExamplesHistory +
        "\n cpUct: " + this.cpUct +
        "\n numberOfMonteCarloSimulations: " + this.nummberOfMonteCarloSimulations;
  }

  public double getNeuralNetworkLearningRate() {
    return neuralNetworkLearningRate;
  }

  public void setNeuralNetworkLearningRate(double neuralNetworkLearningRate) {
    this.neuralNetworkLearningRate = neuralNetworkLearningRate;
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
