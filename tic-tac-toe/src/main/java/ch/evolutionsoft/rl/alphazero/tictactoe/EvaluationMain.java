package ch.evolutionsoft.rl.alphazero.tictactoe;

import java.io.IOException;
import java.util.List;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.common.primitives.Pair;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ch.evolutionsoft.rl.AdversaryLearningConfiguration;
import ch.evolutionsoft.rl.AdversaryLearningSharedHelper;
import ch.evolutionsoft.rl.Game;
import ch.evolutionsoft.rl.alphazero.AdversaryLearning;

public class EvaluationMain {

  private static final Logger log = LoggerFactory.getLogger(EvaluationMain.class);

  public static void main(String[] args) throws IOException {

    AdversaryLearningConfiguration configuration =
        new AdversaryLearningConfiguration.Builder().
        continueTraining(true).
        numberOfIterations(0).
        build();

    ComputationGraph computationGraph = ModelSerializer.restoreComputationGraph(
        configuration.getModelFileName(), true);
    AdversaryLearning adversaryLearning = new AdversaryLearning(
        new TicTacToe(Game.MAX_PLAYER),
        computationGraph,
        configuration);
 
    adversaryLearning.performLearning();
    
    log.info("Empty board probabilities {}", adversaryLearning.getTrainExamplesHistory().get(
        AdversaryLearningSharedHelper.writeStringForArray(TicTacToeConstants.EMPTY_CONVOLUTIONAL_PLAYGROUND)));
    
    evaluateNetwork(computationGraph);
  }

  public static void evaluateNetwork(ComputationGraph graphNetwork) {

    List<Pair<INDArray, INDArray>> allPlaygroundsResults =
        NeuralDataHelper.readAll("/inputs.txt", "/labels.txt");

    List<Pair<INDArray, INDArray>> trainDataSetPairsList =
        TicTacToeNeuralDataConverter.convertMiniMaxPlaygroundLabelsToConvolutionalData(allPlaygroundsResults);

    Pair<INDArray, INDArray> stackedPlaygroundLabels =
        TicTacToeNeuralDataConverter.stackConvolutionalPlaygroundLabels(trainDataSetPairsList);
    DataSet dataSet = new org.nd4j.linalg.dataset.DataSet(
        stackedPlaygroundLabels.getFirst(),
        stackedPlaygroundLabels.getSecond());
    
    INDArray output = graphNetwork.output(dataSet.getFeatures())[0];
    Evaluation eval = new Evaluation(TicTacToeConstants.COLUMN_COUNT);
    eval.eval(dataSet.getLabels(), output);

    if (log.isInfoEnabled()) {
      log.info(eval.stats());
    }
  }

}
