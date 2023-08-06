package ch.evolutionsoft.rl.alphazero.connectfour.model;

import java.beans.PropertyChangeListener;
import java.beans.PropertyChangeSupport;
import java.io.IOException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;

import ch.evolutionsoft.rl.AdversaryLearningConfiguration;
import ch.evolutionsoft.rl.alphazero.*;
import ch.evolutionsoft.rl.alphazero.connectfour.ConnectFour;
import ch.evolutionsoft.rl.alphazero.connectfour.ConnectFourConfiguration;

public class AlphaZeroPlayer extends AbstractPlayer {

  MonteCarloTreeSearch mcts;

  private ConnectFourGame game;

  ConnectFour connectFour;
 
  String modelName;
  
  ComputationGraph model;

  AdversaryLearningConfiguration configuration = ConnectFourConfiguration.getDefaultPlayConfiguration();
  
  double temp = 0.5;
  
  private boolean useMonteCarloSearch;
  private boolean searchInitialized;
  
  private int move;

  private PropertyChangeSupport propertyChangeSupport;

  public AlphaZeroPlayer(int color, String model, boolean useMonteCarloSearch, double temperature) {

    super(color);
    
    try {
      this.model = ModelSerializer.restoreComputationGraph(model);
    } catch (IOException e) {
      System.exit(0);
    }
    this.mcts = new MonteCarloTreeSearch(configuration);
    this.connectFour = new ConnectFour();
    this.modelName = model;
    this.useMonteCarloSearch = useMonteCarloSearch;
    this.temp = temperature;
    this.propertyChangeSupport = new PropertyChangeSupport(this);
  }

  public void addPropertyChangeListener(PropertyChangeListener view) {

    this.propertyChangeSupport.addPropertyChangeListener(view);
  }

  public void reset() {

    this.mcts = new MonteCarloTreeSearch(configuration);
  }

  public void move() {
 
    this.reset();
    
    connectFour = (ConnectFour) connectFour.createNewInstance(
        new BinaryPlayground(game.getPlayGround()), -1);

    this.searchInitialized = true;

    this.propertyChangeSupport.firePropertyChange("searchMove", false, true);

    ExecutorService monteCarloSearchExecutor = Executors.newSingleThreadExecutor();
    monteCarloSearchExecutor.submit(() -> {

      if (useMonteCarloSearch) {
        move = mcts.getActionValues(connectFour, this.temp, this.model.clone()).argMax(0).getInt(0);
      } else {
        INDArray singleBatchBoard = connectFour.getCurrentBoard().reshape(1, 3, PlaygroundConstants.ROW_COUNT, PlaygroundConstants.COLUMN_COUNT);
        INDArray[] neuralNetOutput = this.model.output(singleBatchBoard);
        move = neuralNetOutput[0].argMax(1).getInt(0);
      }

      this.searchInitialized = false;
      this.game.move(move, getColor());
    });
  }

  public void setUseMonteCarloSearch(boolean useMonteCarloSearch) {
    this.useMonteCarloSearch = useMonteCarloSearch;
  }

  public boolean searchInitialized() {
    
    return this.searchInitialized;
  }
  
  public ComputationGraph getModel() {
    
    return this.model;
  }
  
  public void setGame(ConnectFourGame game) {
      
     this.game = game;   
  }
  
  public String toString() {
    
    return this.modelName;
  }
}
