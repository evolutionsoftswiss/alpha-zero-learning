package ch.evolutionsoft.rl.alphazero.connectfour;

import static ch.evolutionsoft.rl.alphazero.connectfour.ConnectFour.*;
import static ch.evolutionsoft.rl.alphazero.connectfour.playground.ArrayPlaygroundConstants.*;

import java.io.IOException;
import java.util.LinkedList;
import java.util.List;

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

    log.info(new NeuralNetMoveEvaluation().evaluateMoves());
    
  }

  ComputationGraph loadConnectFourComputationGraph() throws IOException {

    AdversaryLearningConfiguration adversaryLearningConfiguration = new AdversaryLearningConfiguration.Builder().build();
    
    return ModelSerializer.restoreComputationGraph(adversaryLearningConfiguration.getBestModelFileName());
  }
  
  String evaluateMoves() throws IOException {
    
    this.connectFourComputationGraph = loadConnectFourComputationGraph();

    List<INDArray[]> outputs = getEarlyOpeningOutput();

    String outputsString = "";
    
    for (INDArray[] currentOutputs : outputs) {
      
      outputsString += currentOutputs[0] + System.lineSeparator() + 
          currentOutputs[1] + System.lineSeparator() + System.lineSeparator();
    }
    
    return outputsString;
  }
  
  List<INDArray[]> getEarlyOpeningOutput() {
    
    List<INDArray[]> neuralNetOutputs = new LinkedList<>();

    // Empty Board output
    INDArray neuralNetEmptyBoardInput = createNeuralNetInputSingleBatch(EMPTY_CONVOLUTIONAL_PLAYGROUND.dup());
    INDArray neuralNetEarlyThreatInput = createNeuralNetInputSingleBatch(earlyDoubleThreatPossibilityBoard());

    log.info("Inputs are: {}{}{}{}", System.lineSeparator(), neuralNetEmptyBoardInput, System.lineSeparator(), neuralNetEarlyThreatInput);
    
    INDArray[] valueAndActionOutput1 = this.connectFourComputationGraph.output(neuralNetEmptyBoardInput);
    INDArray[] valueAndActionOutput2 = this.connectFourComputationGraph.output(neuralNetEarlyThreatInput);
    
    neuralNetOutputs.add(valueAndActionOutput1);
    neuralNetOutputs.add(valueAndActionOutput2);
    
    return neuralNetOutputs;
  }
  
  INDArray createNeuralNetInputSingleBatch(INDArray boardInput) {
    
    return boardInput.reshape(1, NUMBER_OF_BOARD_CHANNELS, ArrayPlaygroundConstants.ROW_COUNT, ArrayPlaygroundConstants.COLUMN_COUNT);
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
