package ch.evolutionsoft.rl.alphazero.tictactoe;


import java.util.ArrayList;
import java.util.List;

import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class NeuralDataHelper {

  public static final String LOG_PLACEHOLDER = "{}";
  public static final String IND_ARRAY_VALUE_SEPARATOR = ":";
  public static final String INPUT = "Example Neural Net Input";
  public static final String LABEL = " Label=";

  private NeuralDataHelper() {
    // Hide constructor
  }

  public static List<Pair<INDArray, INDArray>> readAll(String inputPath, String labelPath) {

    List<Pair<INDArray, INDArray>> allPlaygroundsResult = new ArrayList<>();

    INDArray inputs = readInputs(inputPath);
    INDArray labels = readLabels(labelPath);

    for (int row = 0; row < inputs.shape()[0]; row++) {

      allPlaygroundsResult.add(new Pair<>(inputs.getRow(row), labels.getRow(row)));
    }

    return allPlaygroundsResult;
  }

  public static INDArray readInputs(String inputPath) {

    return Nd4j.readTxtString(NeuralDataHelper.class.getResourceAsStream(inputPath));
  }

  public static INDArray readLabels(String labelPath) {

    return Nd4j.readTxtString(NeuralDataHelper.class.getResourceAsStream(labelPath));
  }

}
