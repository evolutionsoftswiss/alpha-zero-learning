package ch.evolutionsoft.rl.alphazero.connectfour;

import static ch.evolutionsoft.rl.alphazero.connectfour.ConnectFour.*;
import static ch.evolutionsoft.rl.alphazero.connectfour.playground.PlaygroundConstants.*;

import java.io.IOException;
import java.util.LinkedList;
import java.util.List;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ch.evolutionsoft.rl.AdversaryLearningConstants;
import ch.evolutionsoft.rl.Game;
import ch.evolutionsoft.rl.alphazero.MonteCarloTreeSearch;

public class NeuralNetMoveEvaluation {

  private static final Logger log = LoggerFactory.getLogger(NeuralNetMoveEvaluation.class);

  ComputationGraph connectFourComputationGraph;
  
  MonteCarloTreeSearch mcts;
  
  public static void main(String[] args) throws IOException {

    if (log.isInfoEnabled()) {
      NeuralNetMoveEvaluation neuralNetMoveEvaluation = new NeuralNetMoveEvaluation();
      log.info(neuralNetMoveEvaluation.evaluateMoves());
    }
  }
  
  ComputationGraph loadConnectFourComputationGraph() throws IOException {
    
    return ModelSerializer.restoreComputationGraph("model.bin");
  }
  
  String evaluateMoves() throws IOException {
    
    this.connectFourComputationGraph = loadConnectFourComputationGraph();
    this.mcts = new MonteCarloTreeSearch(ConnectFourConfiguration.getDefaultPlayConfiguration());

    List<INDArray[]> netOutputs = getEarlyOpeningOutput();

    StringBuilder outputsString = new StringBuilder();
    
    for (INDArray[] currentOutputs : netOutputs) {
      
      outputsString.append(currentOutputs[0]).append(System.lineSeparator()). 
          append(currentOutputs[1]).append(System.lineSeparator()).append(System.lineSeparator());
    }

    List<INDArray> mctsOutputs = getEarlyOpeningMctsResults();   
    for (INDArray currentOutputs : mctsOutputs) {
      
      outputsString.append(currentOutputs).append(System.lineSeparator());
    }
    
    return String.valueOf(outputsString);
  }
  
  List<INDArray[]> getEarlyOpeningOutput() {
    
    List<INDArray[]> neuralNetOutputs = new LinkedList<>();

    // Empty Board output
    INDArray neuralNetEmptyBoardInput = createNeuralNetInputSingleBatch(EMPTY_CONVOLUTIONAL_PLAYGROUND.dup());
    INDArray neuralNetEarlyThreatInput = createNeuralNetInputSingleBatch(earlyDoubleThreatPossibilityBoard());
    INDArray middleOpeningInput = createNeuralNetInputSingleBatch(middleOpeningBoard());
    INDArray badOpeningInput = createNeuralNetInputSingleBatch(badOpeningBoard());

    if (log.isInfoEnabled()) {
      log.info("Inputs are: {}{}{}{}{}{}{}{}",
          System.lineSeparator(),
          neuralNetEmptyBoardInput,
          System.lineSeparator(),
          neuralNetEarlyThreatInput,
          System.lineSeparator(),
          middleOpeningInput,
          System.lineSeparator(),
          badOpeningInput);
    }
      
    INDArray[] valueAndActionOutput1 = this.connectFourComputationGraph.output(neuralNetEmptyBoardInput);
    INDArray[] valueAndActionOutput2 = this.connectFourComputationGraph.output(neuralNetEarlyThreatInput);
    INDArray[] valueAndActionOutput3 = this.connectFourComputationGraph.output(middleOpeningInput);
    INDArray[] valueAndActionOutput4 = this.connectFourComputationGraph.output(badOpeningInput);
    
    neuralNetOutputs.add(valueAndActionOutput1);
    neuralNetOutputs.add(valueAndActionOutput2);
    neuralNetOutputs.add(valueAndActionOutput3);
    neuralNetOutputs.add(valueAndActionOutput4);
    
    return neuralNetOutputs;
  }
  
  List<INDArray> getEarlyOpeningMctsResults() {
    
    List<INDArray> neuralNetOutputs = new LinkedList<>();
 
    Game connectFourEmpty = new ConnectFour();
    Game connectFourEarlyThreat = new ConnectFour();
    connectFourEarlyThreat.makeMove(3, MAX_PLAYER);
    connectFourEarlyThreat.makeMove(3, MIN_PLAYER);
    connectFourEarlyThreat.makeMove(2, MAX_PLAYER);
    
    INDArray valueAndActionOutput1 = this.mcts.getActionValues(connectFourEmpty, 0.0, connectFourComputationGraph);
    INDArray valueAndActionOutput2 = this.mcts.getActionValues(connectFourEarlyThreat, 0.0, connectFourComputationGraph);
    
    neuralNetOutputs.add(valueAndActionOutput1);
    neuralNetOutputs.add(valueAndActionOutput2);
    
    return neuralNetOutputs;
  }
  
  INDArray createNeuralNetInputSingleBatch(INDArray boardInput) {
    
    return boardInput.reshape(1, NUMBER_OF_BOARD_CHANNELS, ROW_COUNT, COLUMN_COUNT);
  }
  
  INDArray middleOpeningBoard() {

    INDArray boardInput = EMPTY_CONVOLUTIONAL_PLAYGROUND.dup();

    // Create horizontal double threat possibility on max next move
    boardInput.putScalar(YELLOW, ROW_COUNT - 1L, 3, AdversaryLearningConstants.ONE);
    boardInput.putSlice(CURRENT_PLAYER_CHANNEL, MINUS_ONES_PLAYGROUND_IMAGE.dup());
    
    return boardInput;
  }
  
  INDArray badOpeningBoard() {

    INDArray boardInput = EMPTY_CONVOLUTIONAL_PLAYGROUND.dup();

    // Create horizontal double threat possibility on max next move
    boardInput.putScalar(YELLOW, ROW_COUNT - 1L, 0, AdversaryLearningConstants.ONE);
    boardInput.putSlice(CURRENT_PLAYER_CHANNEL, MINUS_ONES_PLAYGROUND_IMAGE.dup());
    
    return boardInput;
  }
  
  INDArray earlyDoubleThreatPossibilityBoard() {

    INDArray boardInput = EMPTY_CONVOLUTIONAL_PLAYGROUND.dup();

    // Create horizontal double threat possibility on max next move
    boardInput.putScalar(YELLOW, ROW_COUNT - 1L, 3, AdversaryLearningConstants.ONE);
    boardInput.putScalar(RED, ROW_COUNT - 2L, 3, AdversaryLearningConstants.ONE);
    boardInput.putScalar(YELLOW, ROW_COUNT - 1L, 2, AdversaryLearningConstants.ONE);
    boardInput.putSlice(CURRENT_PLAYER_CHANNEL, MINUS_ONES_PLAYGROUND_IMAGE.dup());
    
    return boardInput;
  }
}
