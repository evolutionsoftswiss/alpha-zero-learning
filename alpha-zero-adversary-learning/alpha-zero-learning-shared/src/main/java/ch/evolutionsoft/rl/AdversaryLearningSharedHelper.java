package ch.evolutionsoft.rl;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
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

public class AdversaryLearningSharedHelper {

  public static final String TRAIN_EXAMPLES_VALUES = "Values";

  Map<String, AdversaryTrainingExample> trainExamplesHistory;

  Map<Integer, Set<String>> trainExampleBoardHashesByIteration;

  private AdversaryLearningConfiguration adversaryLearningConfiguration;

  private static final Logger log = LoggerFactory.getLogger(AdversaryLearningSharedHelper.class);

  public AdversaryLearningSharedHelper(AdversaryLearningConfiguration adversaryLearningConfiguration) {

    this.adversaryLearningConfiguration = adversaryLearningConfiguration;
    
    this.trainExamplesHistory = new HashMap<>(adversaryLearningConfiguration.getMaxTrainExamplesHistory());
    this.trainExampleBoardHashesByIteration = new HashMap<>(
        (adversaryLearningConfiguration.getNumberOfAllAvailableMoves() *
        adversaryLearningConfiguration.getNumberOfAllAvailableMoves() *
        adversaryLearningConfiguration.getNumberOfEpisodesBeforePotentialUpdate()) / 2
     );
  }

  public void loadEarlierTrainingExamples() throws IOException {

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
      
      this.trainExamplesHistory.put(currentAdversaryExample.getBoardString(), currentAdversaryExample);
    }
      
    int size = this.trainExamplesHistory.size();
    log.info("Restored train examples from {} with {} train examples",
         trainExamplesFile,
         size);
  }

  
  public Map<String, AdversaryTrainingExample> getTrainExamplesHistory() {
    
    return this.trainExamplesHistory;
  }
  
  public Map<Integer, Set<String>> getTrainExampleBoardsByIteration() {

    return this.trainExampleBoardHashesByIteration;
  }

  public void initializeTrainExampleBoardsByIterationFromTrainExamplesHistory() {

    this.trainExampleBoardHashesByIteration.putAll(this.trainExamplesHistory.values().stream().collect(
        Collectors.groupingBy(AdversaryTrainingExample::getIteration,
        Collectors.mapping(AdversaryTrainingExample::getBoardString, Collectors.toSet())))
        );
  }
  
  public Map<String, AdversaryTrainingExample> replaceOldTrainingExamplesWithNewActionProbabilities(
      Collection<AdversaryTrainingExample> newExamples) {

    if (newExamples.isEmpty()) {
      
      return Collections.emptyMap();
    }
    
    int replacedNumber = 0;
    Set<String> newIterationBoards = new HashSet<>();
    int currentIteration = newExamples.iterator().next().getIteration();
    Map<String, AdversaryTrainingExample> newExamplesByBoard = new HashMap<>();

    for (AdversaryTrainingExample currentExample : newExamples) {
      
      String currentBoardHashCode = currentExample.getBoardString();
      newIterationBoards.add(currentBoardHashCode);
      AdversaryTrainingExample oldExample = trainExamplesHistory.put(currentBoardHashCode, currentExample);
      newExamplesByBoard.put(currentBoardHashCode, currentExample);

      if (null != oldExample && oldExample.getIteration() != currentIteration) {

        int iteration = oldExample.getIteration();
        Set<String> boardEntriesByOldIteration = trainExampleBoardHashesByIteration.get(iteration);

        if (null != boardEntriesByOldIteration) {

        	boardEntriesByOldIteration.remove(currentBoardHashCode);
	        
	        if (log.isDebugEnabled()) {
	          replacedNumber++;
	        }
        }
      }
    }

    // Handle duplicated iteration numbers
    if (this.trainExampleBoardHashesByIteration.containsKey(currentIteration)) {

      Set<String> earlierIterationBoards = this.trainExampleBoardHashesByIteration.get(currentIteration);
      earlierIterationBoards.addAll(newIterationBoards);
      
    } else {
    
      this.trainExampleBoardHashesByIteration.put(currentIteration, newIterationBoards);
    }
    
    if (log.isDebugEnabled()) {
      
      int listTotalSize = countAllExampleBoardsByIteration();
      log.debug("Updated {} examples with same board from earlier iterations, remaining {} examples are new",
          replacedNumber,
          newExamples.size() - replacedNumber);
      log.debug("New trainExamplesByBoard history map and exampleBoardsByIteration history map size {} and {}",
          trainExamplesHistory.size(),
          listTotalSize);
    }
    
    return newExamplesByBoard;
  }

  public void resizeTrainExamplesHistory(int currentIteration) {

    if (this.adversaryLearningConfiguration.getCurrentMaxTrainExamplesHistory(currentIteration) >=
        this.trainExamplesHistory.size()) {
      
      log.info("New train examples history map iteration {} size {}",
          currentIteration,
          this.trainExamplesHistory.size());
      
      return;
    }

    int previousTrainExamplesSize = this.getTrainExamplesHistory().size();

    SortedSet<Integer> sortedIterationKeys = new TreeSet<>(this.trainExampleBoardHashesByIteration.keySet());
    Iterator<Integer> latestIterationIterator = sortedIterationKeys.iterator();
    
    StringBuilder removedIterations = new StringBuilder();
    while (this.trainExamplesHistory.size() > this.adversaryLearningConfiguration.getCurrentMaxTrainExamplesHistory(currentIteration) &&
        this.trainExampleBoardHashesByIteration.size() > 1) {
      
      Integer remainingOldestIteration = latestIterationIterator.next();
      removedIterations.append(remainingOldestIteration).append(", ");
      Set<String> boardExamplesToBeRemoves = this.trainExampleBoardHashesByIteration.get(remainingOldestIteration);
      
      boardExamplesToBeRemoves.stream().forEach(board -> this.trainExamplesHistory.remove(board));
      this.trainExampleBoardHashesByIteration.remove(remainingOldestIteration);
    }
    
    if (log.isInfoEnabled() && removedIterations.length() > 0) {
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
    for (Set<String> current : trainExampleBoardHashesByIteration.values()) {
      listTotalSize += current.size();
    }
    return listTotalSize;
  }

}
