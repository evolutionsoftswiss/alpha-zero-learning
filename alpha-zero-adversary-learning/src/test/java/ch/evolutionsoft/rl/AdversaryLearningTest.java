package ch.evolutionsoft.rl;

import static org.junit.jupiter.api.Assertions.*;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class AdversaryLearningTest {

  public static final String TEST_TRAIN_EXAMPLES = "testTrainExamples.obj";
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
  }

  @Test
  void testResizeSaveTrainExamplesHistory() throws IOException {

    // Use different dummy keys
    INDArray dummyBoard1 = Nd4j.zeros(1);
    INDArray dummyBoard2 = Nd4j.zeros(2);
    INDArray dummyBoard3 = Nd4j.zeros(3);
    INDArray dummyBoard4 = Nd4j.zeros(4);
    INDArray dummyAction = Nd4j.ones(1);
    
    adversaryLearning.trainExamplesHistory.put(dummyBoard1,
        new AdversaryTrainingExample(dummyBoard1, Game.MAX_PLAYER, dummyAction, 1));
    adversaryLearning.trainExamplesHistory.put(dummyBoard2,
        new AdversaryTrainingExample(dummyBoard2, Game.MAX_PLAYER, dummyAction, 2));
    adversaryLearning.trainExamplesHistory.put(dummyBoard3,
        new AdversaryTrainingExample(dummyBoard3, Game.MAX_PLAYER, dummyAction, 3));
    adversaryLearning.trainExamplesHistory.put(dummyBoard4,
        new AdversaryTrainingExample(dummyBoard4, Game.MAX_PLAYER, dummyAction, 4));
    
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
    INDArray dummyBoard1 = Nd4j.zeros(1);
    INDArray dummyBoard2 = Nd4j.zeros(2);
    INDArray dummyBoard3 = Nd4j.zeros(3);
    INDArray dummyBoard4 = Nd4j.zeros(4);
    INDArray dummyBoard5 = Nd4j.zeros(5);
    INDArray dummyAction = Nd4j.ones(1);
    
    adversaryLearning.trainExamplesHistory.put(dummyBoard1,
        new AdversaryTrainingExample(dummyBoard1, Game.MAX_PLAYER, dummyAction, 1));
    adversaryLearning.trainExamplesHistory.put(dummyBoard2,
        new AdversaryTrainingExample(dummyBoard2, Game.MAX_PLAYER, dummyAction, 2));
    adversaryLearning.trainExamplesHistory.put(dummyBoard3,
        new AdversaryTrainingExample(dummyBoard3, Game.MAX_PLAYER, dummyAction, 2));
    adversaryLearning.trainExamplesHistory.put(dummyBoard4,
        new AdversaryTrainingExample(dummyBoard4, Game.MAX_PLAYER, dummyAction, 3));
    adversaryLearning.trainExamplesHistory.put(dummyBoard5,
        new AdversaryTrainingExample(dummyBoard5, Game.MAX_PLAYER, dummyAction, 3));
    
    adversaryLearning.saveTrainExamplesHistory();
    
    assertEquals(3, adversaryLearning.trainExamplesHistory.size());
    assertFalse(adversaryLearning.trainExamplesHistory.containsKey(dummyBoard1));
    assertTrue(
        adversaryLearning.trainExamplesHistory.containsKey(dummyBoard2) ||
        adversaryLearning.trainExamplesHistory.containsKey(dummyBoard3)
        );
    assertTrue(adversaryLearning.trainExamplesHistory.containsKey(dummyBoard4));
    assertTrue(adversaryLearning.trainExamplesHistory.containsKey(dummyBoard5));
  }

  @Test
  void testResizeWithExactlyMaxExamplesSaveTrainExamplesHistory() throws IOException {

    // Use different dummy keys
    INDArray dummyBoard1 = Nd4j.zeros(1);
    INDArray dummyBoard2 = Nd4j.zeros(2);
    INDArray dummyBoard3 = Nd4j.zeros(3);
    INDArray dummyAction = Nd4j.ones(1);
    
    adversaryLearning.trainExamplesHistory.put(dummyBoard1,
        new AdversaryTrainingExample(dummyBoard1, Game.MAX_PLAYER, dummyAction, 1));
    adversaryLearning.trainExamplesHistory.put(dummyBoard2,
        new AdversaryTrainingExample(dummyBoard2, Game.MAX_PLAYER, dummyAction, 2));
    adversaryLearning.trainExamplesHistory.put(dummyBoard3,
        new AdversaryTrainingExample(dummyBoard3, Game.MAX_PLAYER, dummyAction, 3));
    
    adversaryLearning.saveTrainExamplesHistory();
    
    assertEquals(3, adversaryLearning.trainExamplesHistory.size());
    assertTrue(adversaryLearning.trainExamplesHistory.containsKey(dummyBoard1));
    assertTrue(adversaryLearning.trainExamplesHistory.containsKey(dummyBoard2));
    assertTrue(adversaryLearning.trainExamplesHistory.containsKey(dummyBoard3));
  }

  @Test
  void testResizeLowerMaxExamplesNotNecessarySaveTrainExamplesHistory() throws IOException {

    // Use different dummy keys
    INDArray dummyBoard1 = Nd4j.zeros(1);
    INDArray dummyBoard2 = Nd4j.zeros(2);
    INDArray dummyAction = Nd4j.ones(1);
    
    adversaryLearning.trainExamplesHistory.put(dummyBoard1,
        new AdversaryTrainingExample(dummyBoard1, Game.MAX_PLAYER, dummyAction, 1));
    adversaryLearning.trainExamplesHistory.put(dummyBoard2,
        new AdversaryTrainingExample(dummyBoard2, Game.MAX_PLAYER, dummyAction, 2));
    
    adversaryLearning.saveTrainExamplesHistory();
    
    assertEquals(2, adversaryLearning.trainExamplesHistory.size());
    assertTrue(adversaryLearning.trainExamplesHistory.containsKey(dummyBoard1));
    assertTrue(adversaryLearning.trainExamplesHistory.containsKey(dummyBoard2));
  }

}
