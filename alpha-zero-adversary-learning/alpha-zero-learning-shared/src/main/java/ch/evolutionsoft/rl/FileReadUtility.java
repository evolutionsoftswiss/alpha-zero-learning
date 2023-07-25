package ch.evolutionsoft.rl;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class FileReadUtility {

  public static final String TRAIN_EXAMPLES_VALUES = "Values";

  private FileReadUtility() {
    // hide constructor
  }

  public static Map<String, AdversaryTrainingExample> loadMapFromFile(String trainExamplesFile) throws IOException {
    
    String suffix = "";
    String trainExamplesBasePath = trainExamplesFile;
    if (trainExamplesFile.contains(".")) {
      suffix = trainExamplesFile.substring(trainExamplesFile.lastIndexOf('.'), trainExamplesFile.length());
      int suffixLength = suffix.length();
      trainExamplesBasePath = trainExamplesFile.substring(0, trainExamplesFile.length() - suffixLength);
    }
    INDArray storedBoardKeys;
    try (DataInputStream dataInputStream =
        new DataInputStream(new FileInputStream(trainExamplesFile))) {
      storedBoardKeys = Nd4j.read(dataInputStream);
    }
    INDArray storedValues;
    try (DataInputStream dataInputStream =
        new DataInputStream(new FileInputStream(trainExamplesBasePath + TRAIN_EXAMPLES_VALUES + suffix))) {
      storedValues =  Nd4j.read(dataInputStream);
    }

    long[] actionShape = storedValues.shape();
    int actionIndicesCount = (int) (actionShape[1] - 3);
    Map<String, AdversaryTrainingExample> loadedMap = new HashMap<>();
    
    for (int index = 0; index < storedBoardKeys.shape()[0]; index++) {
      
      INDArray currentBoardKey = storedBoardKeys.slice(index);
      INDArray currentStoredValue = storedValues.getRow(index);
      INDArray actionIndexProbs = Nd4j.zeros(actionIndicesCount);
      
      for (int actionIndex = 0; actionIndex < actionIndicesCount; actionIndex++) {
        
        actionIndexProbs.putScalar(actionIndex, currentStoredValue.getFloat(actionIndex));
      }
      int player = currentStoredValue.getInt(actionIndicesCount);
      float playerValue = currentStoredValue.getFloat(actionIndicesCount + 1L);
      int iterationValue = currentStoredValue.getInt(actionIndicesCount + 2);
      AdversaryTrainingExample currentAdversaryExample =
          new AdversaryTrainingExample(currentBoardKey, player, actionIndexProbs, iterationValue);

      currentAdversaryExample.setCurrentPlayerValue(playerValue);
      
      loadedMap.put(currentAdversaryExample.getBoardString(), currentAdversaryExample);
    }
    return loadedMap;
  }
}
