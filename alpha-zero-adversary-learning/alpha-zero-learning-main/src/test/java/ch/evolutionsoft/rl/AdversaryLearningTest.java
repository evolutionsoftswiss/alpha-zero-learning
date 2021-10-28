package ch.evolutionsoft.rl;

import static org.junit.jupiter.api.Assertions.*;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class AdversaryLearningTest {

  public static final String TEST_TRAIN_EXAMPLES = "testTrainExamples.obj";
  public static final String TEST_TRAIN_EXAMPLES_VALUES = "testTrainExamplesValues.obj";
  AdversaryLearning adversaryLearning;

  @BeforeEach
  void init() {

    adversaryLearning = new AdversaryLearning(
        new TestGame(),
        new TestComputationGraph(),
        new AdversaryLearningConfiguration.Builder().
        maxTrainExamplesHistory(3).
        trainExamplesFileName(TEST_TRAIN_EXAMPLES).
        build());
  }
  
  @AfterEach
  void deleteTestTrainExamples() throws IOException {

    Files.delete(Paths.get(TEST_TRAIN_EXAMPLES));
    Files.delete(Paths.get(TEST_TRAIN_EXAMPLES_VALUES));
    
  }

  @Test
  void testResizeSaveTrainExamplesHistory() throws IOException {

    // Use different dummy keys
    INDArray dummyBoard1 = Nd4j.zeros(3, 6, 7).putScalar(0, 0, 0, 1);
    INDArray dummyBoard2 = Nd4j.zeros(3, 6, 7).putScalar(0, 1, 0, 1);
    INDArray dummyBoard3 = Nd4j.zeros(3, 6, 7).putScalar(0, 2, 0, 1);
    INDArray dummyBoard4 = Nd4j.zeros(3, 6, 7).putScalar(0, 3, 0, 1);
    INDArray dummyAction = Nd4j.ones(7);
    
    AdversaryTrainingExample adversaryTrainingExample1 = new AdversaryTrainingExample(dummyBoard1, Game.MAX_PLAYER, dummyAction, 1);
    adversaryTrainingExample1.setCurrentPlayerValue(0.5f);
    adversaryLearning.trainExamplesHistory.put(dummyBoard1,
        adversaryTrainingExample1);
    AdversaryTrainingExample adversaryTrainingExample2 = new AdversaryTrainingExample(dummyBoard2, Game.MAX_PLAYER, dummyAction, 2);
    adversaryTrainingExample2.setCurrentPlayerValue(0.5f);
    adversaryLearning.trainExamplesHistory.put(dummyBoard2,
        adversaryTrainingExample2);
    AdversaryTrainingExample adversaryTrainingExample3 = new AdversaryTrainingExample(dummyBoard3, Game.MAX_PLAYER, dummyAction, 3);
    adversaryTrainingExample3.setCurrentPlayerValue(0.5f);
    adversaryLearning.trainExamplesHistory.put(dummyBoard3,
        adversaryTrainingExample3);
    AdversaryTrainingExample adversaryTrainingExample4 = new AdversaryTrainingExample(dummyBoard4, Game.MAX_PLAYER, dummyAction, 4);
    adversaryTrainingExample4.setCurrentPlayerValue(0.5f);
    adversaryLearning.trainExamplesHistory.put(dummyBoard4,
        adversaryTrainingExample4);
    
    adversaryLearning.trainExampleBoardsByIteration.put(1, Collections.singleton(dummyBoard1));
    adversaryLearning.trainExampleBoardsByIteration.put(2, Collections.singleton(dummyBoard2));
    adversaryLearning.trainExampleBoardsByIteration.put(3, Collections.singleton(dummyBoard3));
    adversaryLearning.trainExampleBoardsByIteration.put(4, Collections.singleton(dummyBoard4));

    adversaryLearning.saveTrainExamplesHistory();
    
    assertEquals(3, adversaryLearning.trainExamplesHistory.size());
    assertFalse(adversaryLearning.trainExamplesHistory.containsKey(dummyBoard1));
    assertTrue(adversaryLearning.trainExamplesHistory.containsKey(dummyBoard2));
    assertTrue(adversaryLearning.trainExamplesHistory.containsKey(dummyBoard3));
    assertTrue(adversaryLearning.trainExamplesHistory.containsKey(dummyBoard4));
  }

  @Test
  void testResizeSaveTrainExamplesHistoryWithSameIterations() throws IOException {

    // Use different dummy keys
    INDArray dummyBoard1 = Nd4j.zeros(3, 6, 7).putScalar(0, 0, 0, 1);
    INDArray dummyBoard2 = Nd4j.zeros(3, 6, 7).putScalar(0, 1, 0, 1);
    INDArray dummyBoard3 = Nd4j.zeros(3, 6, 7).putScalar(0, 2, 0, 1);
    INDArray dummyBoard4 = Nd4j.zeros(3, 6, 7).putScalar(0, 3, 0, 1);
    INDArray dummyBoard5 = Nd4j.zeros(3, 6, 7).putScalar(0, 4, 0, 1);
    INDArray dummyAction = Nd4j.ones(7);
    
    AdversaryTrainingExample adversaryTrainingExample1 = new AdversaryTrainingExample(dummyBoard1, Game.MAX_PLAYER, dummyAction, 1);
    adversaryTrainingExample1.setCurrentPlayerValue(0.5f);
    adversaryLearning.trainExamplesHistory.put(dummyBoard1,
        adversaryTrainingExample1);
    AdversaryTrainingExample adversaryTrainingExample2 = new AdversaryTrainingExample(dummyBoard2, Game.MAX_PLAYER, dummyAction, 2);
    adversaryTrainingExample2.setCurrentPlayerValue(0.5f);
    adversaryLearning.trainExamplesHistory.put(dummyBoard2,
        adversaryTrainingExample2);
    AdversaryTrainingExample adversaryTrainingExample3 = new AdversaryTrainingExample(dummyBoard3, Game.MAX_PLAYER, dummyAction, 3);
    adversaryTrainingExample3.setCurrentPlayerValue(0.5f);
    adversaryLearning.trainExamplesHistory.put(dummyBoard3,
        adversaryTrainingExample3);
    AdversaryTrainingExample adversaryTrainingExample4 = new AdversaryTrainingExample(dummyBoard4, Game.MAX_PLAYER, dummyAction, 4);
    adversaryTrainingExample4.setCurrentPlayerValue(0.5f);
    adversaryLearning.trainExamplesHistory.put(dummyBoard4,
        adversaryTrainingExample4);
    AdversaryTrainingExample adversaryTrainingExample5 = new AdversaryTrainingExample(dummyBoard5, Game.MAX_PLAYER, dummyAction, 3);
    adversaryTrainingExample5.setCurrentPlayerValue(0.5f);
    adversaryLearning.trainExamplesHistory.put(dummyBoard5,
        adversaryTrainingExample5);
    
    adversaryLearning.trainExampleBoardsByIteration.put(1, Collections.singleton(dummyBoard1));
    adversaryLearning.trainExampleBoardsByIteration.put(2, new HashSet<>(Arrays.asList(dummyBoard2, dummyBoard3)));
    adversaryLearning.trainExampleBoardsByIteration.put(3, new HashSet<>(Arrays.asList(dummyBoard4, dummyBoard5)));

    adversaryLearning.saveTrainExamplesHistory();
    
    assertEquals(2, adversaryLearning.trainExamplesHistory.size());
    assertFalse(adversaryLearning.trainExamplesHistory.containsKey(dummyBoard1));
    assertFalse(adversaryLearning.trainExamplesHistory.containsKey(dummyBoard2));
    assertFalse(adversaryLearning.trainExamplesHistory.containsKey(dummyBoard3));
    assertTrue(adversaryLearning.trainExamplesHistory.containsKey(dummyBoard4));
    assertTrue(adversaryLearning.trainExamplesHistory.containsKey(dummyBoard5));
  }

  @Test
  void testResizeWithExactlyMaxExamplesSaveTrainExamplesHistory() throws IOException {

    // Use different dummy keys
    INDArray dummyBoard1 = Nd4j.zeros(3, 6, 7).putScalar(0, 0, 0, 1);
    INDArray dummyBoard2 = Nd4j.zeros(3, 6, 7).putScalar(0, 1, 0, 1);
    INDArray dummyBoard3 = Nd4j.zeros(3, 6, 7).putScalar(0, 2, 0, 1);
    INDArray dummyAction = Nd4j.ones(7);
    
    AdversaryTrainingExample adversaryTrainingExample1 = new AdversaryTrainingExample(dummyBoard1, Game.MAX_PLAYER, dummyAction, 1);
    adversaryTrainingExample1.setCurrentPlayerValue(0.5f);
    adversaryLearning.trainExamplesHistory.put(dummyBoard1,
        adversaryTrainingExample1);
    AdversaryTrainingExample adversaryTrainingExample2 = new AdversaryTrainingExample(dummyBoard2, Game.MAX_PLAYER, dummyAction, 2);
    adversaryTrainingExample2.setCurrentPlayerValue(0.5f);
    adversaryLearning.trainExamplesHistory.put(dummyBoard2,
        adversaryTrainingExample2);
    AdversaryTrainingExample adversaryTrainingExample3 = new AdversaryTrainingExample(dummyBoard3, Game.MAX_PLAYER, dummyAction, 3);
    adversaryTrainingExample3.setCurrentPlayerValue(0.5f);
    adversaryLearning.trainExamplesHistory.put(dummyBoard3,
        adversaryTrainingExample3);
    
    adversaryLearning.trainExampleBoardsByIteration.put(1, Collections.singleton(dummyBoard1));
    adversaryLearning.trainExampleBoardsByIteration.put(2, Collections.singleton(dummyBoard2));
    adversaryLearning.trainExampleBoardsByIteration.put(3, Collections.singleton(dummyBoard3));

    adversaryLearning.saveTrainExamplesHistory();
    
    assertEquals(3, adversaryLearning.trainExamplesHistory.size());
    assertTrue(adversaryLearning.trainExamplesHistory.containsKey(dummyBoard1));
    assertTrue(adversaryLearning.trainExamplesHistory.containsKey(dummyBoard2));
    assertTrue(adversaryLearning.trainExamplesHistory.containsKey(dummyBoard3));
  }

  @Test
  void testResizeLowerMaxExamplesNotNecessarySaveTrainExamplesHistory() throws IOException {

    // Use different dummy keys
    INDArray dummyBoard1 = Nd4j.zeros(3, 6, 7).putScalar(0, 0, 0, 1);
    INDArray dummyBoard2 = Nd4j.zeros(3, 6, 7).putScalar(0, 1, 0, 1);
    INDArray dummyAction = Nd4j.ones(7);
    
    AdversaryTrainingExample adversaryTrainingExample1 = new AdversaryTrainingExample(dummyBoard1, Game.MAX_PLAYER, dummyAction, 1);
    adversaryTrainingExample1.setCurrentPlayerValue(0.5f);
    adversaryLearning.trainExamplesHistory.put(dummyBoard1,
        adversaryTrainingExample1);
    AdversaryTrainingExample adversaryTrainingExample2 = new AdversaryTrainingExample(dummyBoard2, Game.MAX_PLAYER, dummyAction, 2);
    adversaryTrainingExample2.setCurrentPlayerValue(0.5f);
    adversaryLearning.trainExamplesHistory.put(dummyBoard2,
        adversaryTrainingExample2);
    
    adversaryLearning.trainExampleBoardsByIteration.put(1, Collections.singleton(dummyBoard1));
    adversaryLearning.trainExampleBoardsByIteration.put(2, Collections.singleton(dummyBoard2));

    adversaryLearning.saveTrainExamplesHistory();
    
    assertEquals(2, adversaryLearning.trainExamplesHistory.size());
    assertTrue(adversaryLearning.trainExamplesHistory.containsKey(dummyBoard1));
    assertTrue(adversaryLearning.trainExamplesHistory.containsKey(dummyBoard2));
  }

}
