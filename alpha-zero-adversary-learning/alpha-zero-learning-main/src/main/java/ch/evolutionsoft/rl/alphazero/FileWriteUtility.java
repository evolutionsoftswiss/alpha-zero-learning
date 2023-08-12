package ch.evolutionsoft.rl.alphazero;

import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Map;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import ch.evolutionsoft.rl.AdversaryTrainingExample;

public class FileWriteUtility {

  public static final int SEVEN_DIGITS = 7;

  private FileWriteUtility() {
    // hide constructor
  }
  
  public static void writeTrainExamplesToFiles(String trainExamplesKeyPath, String trainExamplesValuesPath,
      Map<String, AdversaryTrainingExample> sourceMap) throws IOException {

    if (!sourceMap.isEmpty()) {
      AdversaryTrainingExample example = sourceMap.values().iterator().next();

      long[] boardShape = example.getBoard().shape();
      long[] actionShape = example.getActionIndexProbabilities().shape();
      
      INDArray allBoardsKey = Nd4j.zeros(sourceMap.size(), boardShape[0], boardShape[1], boardShape[2]);
      INDArray allValues = Nd4j.zeros(sourceMap.size(), actionShape[0] + 3);

      int exampleNumber = 0;
      for (Map.Entry<String, AdversaryTrainingExample> currentExampleEntry : sourceMap.entrySet()) {

        allBoardsKey.putSlice(exampleNumber, currentExampleEntry.getValue().getBoard());
        INDArray valueNDArray = Nd4j.zeros(actionShape[0] + 3);

        AdversaryTrainingExample value = currentExampleEntry.getValue();
        INDArray actionIndexProbabilities = value.getActionIndexProbabilities();
        for (int actionIndex = 0; actionIndex <= actionShape[0] - 1; actionIndex++) {

          valueNDArray.putScalar(actionIndex, actionIndexProbabilities.getFloat(actionIndex));
        }
        valueNDArray.putScalar(actionShape[0], value.getCurrentPlayer());
        valueNDArray.putScalar(actionShape[0] + 1, value.getCurrentPlayerValue());
        valueNDArray.putScalar(actionShape[0] + 2, value.getIteration());
        allValues.putSlice(exampleNumber, valueNDArray);

        exampleNumber++;
      }

      try (DataOutputStream dataOutputStream = new DataOutputStream(new FileOutputStream(trainExamplesKeyPath))) {

        Nd4j.write(allBoardsKey, dataOutputStream);
      }

      try (DataOutputStream dataOutputStream = new DataOutputStream(new FileOutputStream(trainExamplesValuesPath))) {

        Nd4j.write(allValues, dataOutputStream);
      }
    }
  }

  public static StringBuilder prependZeros(int iteration) {

    int prependingZeros = SEVEN_DIGITS - String.valueOf(iteration).length();

    StringBuilder prependedZeros = new StringBuilder();
    for (int n = 1; n <= prependingZeros; n++) {
      prependedZeros.append('0');
    }
    return prependedZeros;
  }
}
