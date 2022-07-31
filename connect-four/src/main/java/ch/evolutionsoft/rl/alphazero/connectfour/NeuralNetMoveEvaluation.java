package ch.evolutionsoft.rl.alphazero.connectfour;

import static ch.evolutionsoft.rl.alphazero.connectfour.ConnectFour.*;
import static ch.evolutionsoft.rl.alphazero.connectfour.playground.ArrayPlaygroundConstants.*;

import java.io.IOException;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ch.evolutionsoft.rl.AdversaryLearningConfiguration;
import ch.evolutionsoft.rl.AdversaryLearningConstants;
import ch.evolutionsoft.rl.alphazero.connectfour.playground.ArrayPlaygroundConstants;

public class NeuralNetMoveEvaluation {

  private static final Logger log = LoggerFactory.getLogger(NeuralNetMoveEvaluation.class);

  ComputationGraph connectFourComputationGraph;
  
  public static void main(String[] args) throws IOException {

    if (log.isInfoEnabled()) {
      NeuralNetMoveEvaluation neuralNetMoveEvaluation = new NeuralNetMoveEvaluation();
      log.info(neuralNetMoveEvaluation.evaluateMoves());
    }
  }
  
  ComputationGraph loadConnectFourComputationGraph() throws IOException {

    AdversaryLearningConfiguration adversaryLearningConfiguration = new AdversaryLearningConfiguration.Builder().build();
    
    return ModelSerializer.restoreComputationGraph(adversaryLearningConfiguration.getBestModelFileName());
  }
  
  String evaluateMoves() throws IOException {
    
    this.connectFourComputationGraph = loadConnectFourComputationGraph();

    List<INDArray[]> outputs = getEarlyOpeningOutput();

    StringBuilder outputsString = new StringBuilder();
    
    for (INDArray[] currentOutputs : outputs) {
      
      outputsString.append(currentOutputs[0]).append(System.lineSeparator()). 
          append(currentOutputs[1]).append(System.lineSeparator()).append(System.lineSeparator());
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
  
  INDArray createNeuralNetInputSingleBatch(INDArray boardInput) {
    
    return boardInput.reshape(1, NUMBER_OF_BOARD_CHANNELS, ArrayPlaygroundConstants.ROW_COUNT, ArrayPlaygroundConstants.COLUMN_COUNT);
  }
  
  INDArray middleOpeningBoard() {

    INDArray boardInput = EMPTY_CONVOLUTIONAL_PLAYGROUND.dup();

    // Create horizontal double threat possibility on max next move
    boardInput.putScalar(YELLOW, ArrayPlaygroundConstants.ROW_COUNT - 1L, 3, AdversaryLearningConstants.ONE);
    boardInput.putRow(CURRENT_PLAYER_CHANNEL, MINUS_ONES_PLAYGROUND_IMAGE.dup());
    
    return boardInput;
  }
  
  INDArray badOpeningBoard() {

    INDArray boardInput = EMPTY_CONVOLUTIONAL_PLAYGROUND.dup();

    // Create horizontal double threat possibility on max next move
    boardInput.putScalar(YELLOW, ArrayPlaygroundConstants.ROW_COUNT - 1L, 0, AdversaryLearningConstants.ONE);
    boardInput.putRow(CURRENT_PLAYER_CHANNEL, MINUS_ONES_PLAYGROUND_IMAGE.dup());
    
    return boardInput;
  }
  
  INDArray earlyDoubleThreatPossibilityBoard() {

    INDArray boardInput = EMPTY_CONVOLUTIONAL_PLAYGROUND.dup();

    // Create horizontal double threat possibility on max next move
    boardInput.putScalar(YELLOW, ArrayPlaygroundConstants.ROW_COUNT - 1L, 3, AdversaryLearningConstants.ONE);
    boardInput.putScalar(RED, ArrayPlaygroundConstants.ROW_COUNT - 2L, 3, AdversaryLearningConstants.ONE);
    boardInput.putScalar(YELLOW, ArrayPlaygroundConstants.ROW_COUNT - 1L, 2, AdversaryLearningConstants.ONE);
    boardInput.putRow(CURRENT_PLAYER_CHANNEL, MINUS_ONES_PLAYGROUND_IMAGE.dup());
    
    return boardInput;
  }
}
