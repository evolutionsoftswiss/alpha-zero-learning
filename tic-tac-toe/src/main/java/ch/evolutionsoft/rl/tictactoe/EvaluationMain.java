package ch.evolutionsoft.rl.tictactoe;

import static ch.evolutionsoft.net.game.tictactoe.TicTacToeConstants.COLUMN_COUNT;

import java.io.IOException;
import java.util.List;
import java.util.Map;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.common.primitives.Pair;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ch.evolutionsoft.net.game.NeuralDataHelper;
import ch.evolutionsoft.net.game.tictactoe.TicTacToeConstants;
import ch.evolutionsoft.net.game.tictactoe.TicTacToeNeuralDataConverter;
import ch.evolutionsoft.rl.AdversaryLearning;
import ch.evolutionsoft.rl.AdversaryLearningConfiguration;
import ch.evolutionsoft.rl.AdversaryTrainingExample;
import ch.evolutionsoft.rl.Game;

public class EvaluationMain {

  private static final Logger log = LoggerFactory.getLogger(EvaluationMain.class);

  public static void main(String[] args) throws IOException {

    ComputationGraph computationGraph1 = ModelSerializer.restoreComputationGraph("bestmodel.bin", true);

    AdversaryLearning al = new AdversaryLearning(
        new TicTacToe(Game.MAX_PLAYER),
        computationGraph1,
        new AdversaryLearningConfiguration.Builder().iterationStart(201).build());
    Map<INDArray, AdversaryTrainingExample> examples = al.loadEarlierTrainingExamples("trainExamples.obj");
 
    log.info("Empty field probabilities {}", examples.get(TicTacToeConstants.EMPTY_CONVOLUTIONAL_PLAYGROUND));
    
    evaluateNetwork(computationGraph1);
  }

  public static void evaluateNetwork(ComputationGraph graphNetwork) {

    List<Pair<INDArray, INDArray>> allPlaygroundsResults =
        NeuralDataHelper.readAll("/inputs.txt", "/labels.txt");

    List<Pair<INDArray, INDArray>> trainDataSetPairsList =
        TicTacToeNeuralDataConverter.convertMiniMaxPlaygroundLabelsToConvolutionalData(allPlaygroundsResults);

    Pair<INDArray, INDArray> stackedPlaygroundLabels =
        TicTacToeNeuralDataConverter.stackConvolutionalPlaygroundLabels(trainDataSetPairsList);
    DataSet dataSet = new org.nd4j.linalg.dataset.DataSet(stackedPlaygroundLabels.getFirst(), stackedPlaygroundLabels.getSecond());
    
    INDArray output = graphNetwork.output(dataSet.getFeatures())[0];
    Evaluation eval = new Evaluation(COLUMN_COUNT);
    eval.eval(dataSet.getLabels(), output);

    if (log.isInfoEnabled()) {
      log.info(eval.stats());
    }
  }

}
