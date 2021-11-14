package ch.evolutionsoft.rl;

import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import java.util.SortedSet;
import java.util.TreeSet;
import java.util.stream.Collectors;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.string.NDArrayStrings;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonSetter;

public class AdversaryLearningSharedHelper {

  public static final String TRAIN_EXAMPLES_VALUES = "Values";

  Map<INDArray, AdversaryTrainingExample> trainExamplesHistory;

  Map<Integer, Set<INDArray>> trainExampleBoardsByIteration;

  private AdversaryLearningConfiguration adversaryLearningConfiguration;

  private static final Logger log = LoggerFactory.getLogger(AdversaryLearningSharedHelper.class);

  public AdversaryLearningSharedHelper(AdversaryLearningConfiguration adversaryLearningConfiguration) {

    this.adversaryLearningConfiguration = adversaryLearningConfiguration;
    
    this.trainExamplesHistory = new HashMap<>(adversaryLearningConfiguration.getMaxTrainExamplesHistory());
    this.trainExampleBoardsByIteration = new HashMap<>(
        (adversaryLearningConfiguration.getNumberOfAllAvailableMoves() *
        adversaryLearningConfiguration.getNumberOfAllAvailableMoves() *
        adversaryLearningConfiguration.getNumberOfEpisodesBeforePotentialUpdate()) / 2
     );
  }

  public void loadEarlierTrainingExamples(boolean restoreTrainingExamples) throws IOException {

    if (restoreTrainingExamples) {

      log.info("Restoring trainExamplesByBoard history map, this may take a while...");
      
      String trainExamplesFile = adversaryLearningConfiguration.getTrainExamplesFileName();
      loadMapFromFile(trainExamplesFile);

      log.info("Restoring exampleBoardsByIteration from trainExamplesByBoard map...");
      
      this.initializeTrainExampleBoardsByIterationFromTrainExamplesHistory();

      log.info("Train examples maps restored from {}", trainExamplesFile);
      log.info("trainExamplesByBoard map has {} restored AdversaryTrainingExamples entries",
          this.getTrainExamplesHistory().size());
      log.info("exampleBoardsByIteration map has {} restored Set of boards entries with total {} examples",
          this.getTrainExampleBoardsByIteration().size(),
          this.countAllExampleBoardsByIteration());
    }
  }

  void loadMapFromFile(String trainExamplesFile) throws IOException {
    
    String suffix = "";
    String trainExamplesBasePath = trainExamplesFile;
    if (adversaryLearningConfiguration.getTrainExamplesFileName().contains(".")) {
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
    for (int index = 0; index < storedBoardKeys.shape() [0]; index++) {
      
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
      
      this.trainExamplesHistory.put(currentBoardKey, currentAdversaryExample);
    }
      
    int size = this.trainExamplesHistory.size();
    log.info("Restored train examples from {} with {} train examples",
         trainExamplesFile,
         size);
  }

  
  public Map<INDArray, AdversaryTrainingExample> getTrainExamplesHistory() {
    
    return this.trainExamplesHistory;
  }

  @JsonProperty("trainExamplesHistory")
  public Map<String, AdversaryTrainingExample> getTrainExamplesHistoryStringKeys() {
    
    Map<String, AdversaryTrainingExample> jsonConvertedMap = new HashMap<>();
    
    for (Map.Entry<INDArray, AdversaryTrainingExample> originalEntry : this.trainExamplesHistory.entrySet()) {
      
      jsonConvertedMap.put(writeStringForArray(originalEntry.getKey()), originalEntry.getValue());
    }
    
    return jsonConvertedMap;
  }

  @JsonSetter
  public void setTrainExamplesHistory(Map<String, AdversaryTrainingExample> jsonTrainExamplesHistory) {
    
    this.trainExamplesHistory.clear();
    
    for (Map.Entry<String, AdversaryTrainingExample> jsonEntry : jsonTrainExamplesHistory.entrySet()) {
      
      this.trainExamplesHistory.put(
          Nd4j.readTxtString(new ByteArrayInputStream(jsonEntry.getKey().getBytes())),
          jsonEntry.getValue());
    }
  }
  
  public Map<Integer, Set<INDArray>> getTrainExampleBoardsByIteration() {

    return this.trainExampleBoardsByIteration;
  }
  
  public void putTrainExampleBoardsByIteration(int iteration, Set<INDArray> boards) {
    
    this.trainExampleBoardsByIteration.put(iteration, boards);
  }

  public void initializeTrainExampleBoardsByIterationFromTrainExamplesHistory() {

    this.trainExampleBoardsByIteration.putAll(this.trainExamplesHistory.values().stream().collect(
        Collectors.groupingBy(AdversaryTrainingExample::getIteration,
        Collectors.mapping(AdversaryTrainingExample::getBoard, Collectors.toSet())))
        );
  }
  
  public void replaceOldTrainingExamplesWithNewActionProbabilities(
      Collection<AdversaryTrainingExample> newExamples) {

    int replacedNumber = 0;
    Set<INDArray> newIterationBoards = new HashSet<>();
    int currentIteration = newExamples.iterator().next().getIteration();

    for (AdversaryTrainingExample currentExample : newExamples) {
      
      INDArray currentBoard = currentExample.getBoard();
      newIterationBoards.add(currentBoard);
      AdversaryTrainingExample oldExample = trainExamplesHistory.put(currentBoard, currentExample);
      
      if (null != oldExample && oldExample.getIteration() != currentIteration) {
        Set<INDArray> boardEntriesByOldIteration = trainExampleBoardsByIteration.get(oldExample.getIteration());
        boardEntriesByOldIteration.remove(currentBoard);
        
        if (log.isDebugEnabled()) {
          replacedNumber++;
        }
      }
    }

    trainExampleBoardsByIteration.put(currentIteration, newIterationBoards);
    
    if (log.isDebugEnabled()) {
      
      int listTotalSize = countAllExampleBoardsByIteration();
      log.debug("Updated {} examples with same board from earlier iterations, remaining {} examples are new",
          replacedNumber,
          newExamples.size() - replacedNumber);
      log.debug("New trainExamplesByBoard history map and exampleBoardsByIteration history map size {} and {}",
          trainExamplesHistory.size(),
          listTotalSize);
    }
  }

  public void resizeTrainExamplesHistory() {

    if (this.adversaryLearningConfiguration.getMaxTrainExamplesHistory() >=
        this.trainExamplesHistory.size()) {
      
      log.info("New train examples history map size {}",
          this.trainExamplesHistory.size());
      
      return;
    }

    int previousTrainExamplesSize = this.getTrainExamplesHistory().size();

    SortedSet<Integer> sortedIterationKeys = new TreeSet<>(this.trainExampleBoardsByIteration.keySet());
    Iterator<Integer> latestIterationIterator = sortedIterationKeys.iterator();
    
    StringBuilder removedIterations = new StringBuilder();
    while (this.trainExamplesHistory.size() > this.adversaryLearningConfiguration.getMaxTrainExamplesHistory()) {
      
      Integer remainingOldestIteration = latestIterationIterator.next();
      removedIterations.append(remainingOldestIteration).append(", ");
      Set<INDArray> boardExamplesToBeRemoves = this.trainExampleBoardsByIteration.get(remainingOldestIteration);
      
      boardExamplesToBeRemoves.stream().forEach(board -> this.trainExamplesHistory.remove(board));
      this.trainExampleBoardsByIteration.remove(remainingOldestIteration);
    }
    
    if (log.isInfoEnabled()) {
      log.info("Board examples from iteration[s] {} removed", removedIterations.substring(0, removedIterations.length() - 2));
      
      log.info(
          "Oldest from {} examples history removed to keep {} examples",
          previousTrainExamplesSize,
          this.trainExamplesHistory.size());
    }
  }
  
  public static String writeStringForArray(INDArray write) {
    if(write.isView() || !Shape.hasDefaultStridesForShape(write))
      write = write.dup();

    String format = "0.00000000E0";
  
    return "{\n" +
            "\"filefrom\": \"dl4j\",\n" +
            "\"ordering\": \"" + write.ordering() + "\",\n" +
            "\"shape\":\t" + Arrays.toString(write.shape()) + ",\n" +
            "\"data\":\n" +
            new NDArrayStrings(",", format).format(write, false) +
            "\n}\n";
  }

  public int countAllExampleBoardsByIteration() {

    int listTotalSize = 0;
    for (Set<INDArray> current : trainExampleBoardsByIteration.values()) {
      listTotalSize += current.size();
    }
    return listTotalSize;
  }

}
