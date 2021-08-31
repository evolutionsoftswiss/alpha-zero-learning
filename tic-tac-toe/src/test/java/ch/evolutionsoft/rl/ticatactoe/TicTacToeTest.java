package ch.evolutionsoft.rl.ticatactoe;

import static org.junit.jupiter.api.Assertions.*;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import ch.evolutionsoft.rl.AdversaryLearningConstants;
import ch.evolutionsoft.rl.AdversaryTrainingExample;
import ch.evolutionsoft.rl.Game;
import ch.evolutionsoft.rl.tictactoe.TicTacToe;
import ch.evolutionsoft.rl.tictactoe.TicTacToeConstants;

public class TicTacToeTest {

  TicTacToe ticTacToe = new TicTacToe(Game.MAX_PLAYER);
  
  @Test
  public void testSymmetriesWithNoDifferentSymmetry() {

    INDArray middleBoard = TicTacToeConstants.EMPTY_CONVOLUTIONAL_PLAYGROUND.dup();
    middleBoard.putScalar(TicTacToeConstants.MAX_PLAYER_CHANNEL, 1, 1, AdversaryLearningConstants.ONE);
    
    List<AdversaryTrainingExample> trainingExampleSymmetries =
        ticTacToe.getSymmetries(middleBoard, Nd4j.zeros(TicTacToeConstants.COLUMN_COUNT), Game.MAX_PLAYER, 0);
    
    AdversaryTrainingExample symmetryExample =
        new AdversaryTrainingExample(middleBoard, Game.MAX_PLAYER, Nd4j.zeros(TicTacToeConstants.COLUMN_COUNT), 0);
    
    Set<AdversaryTrainingExample> uniqueSymmetries = new HashSet<>();
    uniqueSymmetries.add(symmetryExample);
    
    for (int index = 0; index < trainingExampleSymmetries.size(); index++) {
      
      AdversaryTrainingExample adversaryTrainingExample = trainingExampleSymmetries.get(index);
      uniqueSymmetries.add(adversaryTrainingExample);
    }
    
    assertSame(1, uniqueSymmetries.size());

    assertTrue(uniqueSymmetries.contains(symmetryExample));
  }
  
  @Test
  public void testSymmetriesWith4Symmetries() {
    
    INDArray topLeftMiddleBoard = TicTacToeConstants.EMPTY_CONVOLUTIONAL_PLAYGROUND.dup();
    topLeftMiddleBoard.putScalar(TicTacToeConstants.MAX_PLAYER_CHANNEL, 1, 1, AdversaryLearningConstants.ONE);
    topLeftMiddleBoard.putScalar(TicTacToeConstants.MIN_PLAYER_CHANNEL, 0, 0, AdversaryLearningConstants.ONE);
    
    List<AdversaryTrainingExample> trainingExampleSymmetries =
        ticTacToe.getSymmetries(topLeftMiddleBoard, Nd4j.zeros(TicTacToeConstants.COLUMN_COUNT), Game.MAX_PLAYER, 0);
    
    AdversaryTrainingExample symmetry0Example =
        new AdversaryTrainingExample(topLeftMiddleBoard, Game.MAX_PLAYER, Nd4j.zeros(TicTacToeConstants.COLUMN_COUNT), 0);
    
    Set<AdversaryTrainingExample> uniqueSymmetries = new HashSet<>();
    uniqueSymmetries.add(symmetry0Example);
    
    for (int index = 0; index < trainingExampleSymmetries.size(); index++) {
      
      AdversaryTrainingExample adversaryTrainingExample = trainingExampleSymmetries.get(index);
      uniqueSymmetries.add(adversaryTrainingExample);
    }
    
    assertSame(4, uniqueSymmetries.size());
    
    INDArray symmetry1 = TicTacToeConstants.EMPTY_CONVOLUTIONAL_PLAYGROUND.dup();
    symmetry1.putScalar(TicTacToeConstants.MAX_PLAYER_CHANNEL, 1, 1, AdversaryLearningConstants.ONE);
    symmetry1.putScalar(TicTacToeConstants.MIN_PLAYER_CHANNEL, 0, 2, AdversaryLearningConstants.ONE);
  
    INDArray symmetry2 = TicTacToeConstants.EMPTY_CONVOLUTIONAL_PLAYGROUND.dup();
    symmetry2.putScalar(TicTacToeConstants.MAX_PLAYER_CHANNEL, 1, 1, AdversaryLearningConstants.ONE);
    symmetry2.putScalar(TicTacToeConstants.MIN_PLAYER_CHANNEL, 2, 2, AdversaryLearningConstants.ONE);
    
    INDArray symmetry3 = TicTacToeConstants.EMPTY_CONVOLUTIONAL_PLAYGROUND.dup();
    symmetry3.putScalar(TicTacToeConstants.MAX_PLAYER_CHANNEL, 1, 1, AdversaryLearningConstants.ONE);
    symmetry3.putScalar(TicTacToeConstants.MIN_PLAYER_CHANNEL, 2, 0, AdversaryLearningConstants.ONE);
    
    assertTrue(uniqueSymmetries.contains(symmetry0Example));
    assertTrue(uniqueSymmetries.contains(
        new AdversaryTrainingExample(symmetry1, Game.MAX_PLAYER, Nd4j.zeros(TicTacToeConstants.COLUMN_COUNT), 0)));
    assertTrue(uniqueSymmetries.contains(
        new AdversaryTrainingExample(symmetry2, Game.MAX_PLAYER, Nd4j.zeros(TicTacToeConstants.COLUMN_COUNT), 0)));
    assertTrue(uniqueSymmetries.contains(
        new AdversaryTrainingExample(symmetry3, Game.MAX_PLAYER, Nd4j.zeros(TicTacToeConstants.COLUMN_COUNT), 0)));
  }
  
  @Test
  public void testSymmetriesWith8Symmetries() {
    
    INDArray lBoard = TicTacToeConstants.EMPTY_CONVOLUTIONAL_PLAYGROUND.dup();
    lBoard.putScalar(TicTacToeConstants.MAX_PLAYER_CHANNEL, 1, 1, AdversaryLearningConstants.ONE);
    lBoard.putScalar(TicTacToeConstants.MAX_PLAYER_CHANNEL, 2, 1, AdversaryLearningConstants.ONE);
    lBoard.putScalar(TicTacToeConstants.MIN_PLAYER_CHANNEL, 0, 0, AdversaryLearningConstants.ONE);
    lBoard.putScalar(TicTacToeConstants.MIN_PLAYER_CHANNEL, 0, 1, AdversaryLearningConstants.ONE);
    
    List<AdversaryTrainingExample> trainingExampleSymmetries =
        ticTacToe.getSymmetries(lBoard, Nd4j.zeros(TicTacToeConstants.COLUMN_COUNT), Game.MAX_PLAYER, 0);
    
    AdversaryTrainingExample symmetry0Example =
        new AdversaryTrainingExample(lBoard, Game.MAX_PLAYER, Nd4j.zeros(TicTacToeConstants.COLUMN_COUNT), 0);
    
    Set<AdversaryTrainingExample> uniqueSymmetries = new HashSet<>();
    uniqueSymmetries.add(symmetry0Example);
    
    for (int index = 0; index < trainingExampleSymmetries.size(); index++) {
      
      AdversaryTrainingExample adversaryTrainingExample = trainingExampleSymmetries.get(index);
      uniqueSymmetries.add(adversaryTrainingExample);
    }
    
    assertSame(8, uniqueSymmetries.size());
    
    INDArray symmetry1 = TicTacToeConstants.EMPTY_CONVOLUTIONAL_PLAYGROUND.dup();
    symmetry1.putScalar(TicTacToeConstants.MAX_PLAYER_CHANNEL, 1, 1, AdversaryLearningConstants.ONE);
    symmetry1.putScalar(TicTacToeConstants.MAX_PLAYER_CHANNEL, 0, 1, AdversaryLearningConstants.ONE);
    symmetry1.putScalar(TicTacToeConstants.MIN_PLAYER_CHANNEL, 2, 0, AdversaryLearningConstants.ONE);
    symmetry1.putScalar(TicTacToeConstants.MIN_PLAYER_CHANNEL, 2, 1, AdversaryLearningConstants.ONE);
  
    INDArray symmetry2 = TicTacToeConstants.EMPTY_CONVOLUTIONAL_PLAYGROUND.dup();
    symmetry2.putScalar(TicTacToeConstants.MAX_PLAYER_CHANNEL, 1, 1, AdversaryLearningConstants.ONE);
    symmetry2.putScalar(TicTacToeConstants.MAX_PLAYER_CHANNEL, 1, 0, AdversaryLearningConstants.ONE);
    symmetry2.putScalar(TicTacToeConstants.MIN_PLAYER_CHANNEL, 1, 2, AdversaryLearningConstants.ONE);
    symmetry2.putScalar(TicTacToeConstants.MIN_PLAYER_CHANNEL, 2, 2, AdversaryLearningConstants.ONE);
    
    INDArray symmetry3 = TicTacToeConstants.EMPTY_CONVOLUTIONAL_PLAYGROUND.dup();
    symmetry3.putScalar(TicTacToeConstants.MAX_PLAYER_CHANNEL, 1, 1, AdversaryLearningConstants.ONE);
    symmetry3.putScalar(TicTacToeConstants.MAX_PLAYER_CHANNEL, 1, 0, AdversaryLearningConstants.ONE);
    symmetry3.putScalar(TicTacToeConstants.MIN_PLAYER_CHANNEL, 1, 2, AdversaryLearningConstants.ONE);
    symmetry3.putScalar(TicTacToeConstants.MIN_PLAYER_CHANNEL, 0, 2, AdversaryLearningConstants.ONE);
    
    INDArray symmetry4 = TicTacToeConstants.EMPTY_CONVOLUTIONAL_PLAYGROUND.dup();
    symmetry4.putScalar(TicTacToeConstants.MAX_PLAYER_CHANNEL, 1, 1, AdversaryLearningConstants.ONE);
    symmetry4.putScalar(TicTacToeConstants.MAX_PLAYER_CHANNEL, 1, 2, AdversaryLearningConstants.ONE);
    symmetry4.putScalar(TicTacToeConstants.MIN_PLAYER_CHANNEL, 1, 0, AdversaryLearningConstants.ONE);
    symmetry4.putScalar(TicTacToeConstants.MIN_PLAYER_CHANNEL, 2, 0, AdversaryLearningConstants.ONE);
    
    INDArray symmetry5 = TicTacToeConstants.EMPTY_CONVOLUTIONAL_PLAYGROUND.dup();
    symmetry5.putScalar(TicTacToeConstants.MAX_PLAYER_CHANNEL, 1, 1, AdversaryLearningConstants.ONE);
    symmetry5.putScalar(TicTacToeConstants.MAX_PLAYER_CHANNEL, 0, 1, AdversaryLearningConstants.ONE);
    symmetry5.putScalar(TicTacToeConstants.MIN_PLAYER_CHANNEL, 2, 1, AdversaryLearningConstants.ONE);
    symmetry5.putScalar(TicTacToeConstants.MIN_PLAYER_CHANNEL, 2, 2, AdversaryLearningConstants.ONE);
    
    INDArray symmetry6 = TicTacToeConstants.EMPTY_CONVOLUTIONAL_PLAYGROUND.dup();
    symmetry6.putScalar(TicTacToeConstants.MAX_PLAYER_CHANNEL, 1, 1, AdversaryLearningConstants.ONE);
    symmetry6.putScalar(TicTacToeConstants.MAX_PLAYER_CHANNEL, 1, 2, AdversaryLearningConstants.ONE);
    symmetry6.putScalar(TicTacToeConstants.MIN_PLAYER_CHANNEL, 0, 0, AdversaryLearningConstants.ONE);
    symmetry6.putScalar(TicTacToeConstants.MIN_PLAYER_CHANNEL, 1, 0, AdversaryLearningConstants.ONE);
    
    INDArray symmetry7 = TicTacToeConstants.EMPTY_CONVOLUTIONAL_PLAYGROUND.dup();
    symmetry7.putScalar(TicTacToeConstants.MAX_PLAYER_CHANNEL, 1, 1, AdversaryLearningConstants.ONE);
    symmetry7.putScalar(TicTacToeConstants.MAX_PLAYER_CHANNEL, 1, 2, AdversaryLearningConstants.ONE);
    symmetry7.putScalar(TicTacToeConstants.MIN_PLAYER_CHANNEL, 2, 0, AdversaryLearningConstants.ONE);
    symmetry7.putScalar(TicTacToeConstants.MIN_PLAYER_CHANNEL, 1, 0, AdversaryLearningConstants.ONE);
    
    assertTrue(uniqueSymmetries.contains(symmetry0Example));
    assertTrue(uniqueSymmetries.contains(
        new AdversaryTrainingExample(symmetry1, Game.MAX_PLAYER, Nd4j.zeros(TicTacToeConstants.COLUMN_COUNT), 0)));
    assertTrue(uniqueSymmetries.contains(
        new AdversaryTrainingExample(symmetry2, Game.MAX_PLAYER, Nd4j.zeros(TicTacToeConstants.COLUMN_COUNT), 0)));
    assertTrue(uniqueSymmetries.contains(
        new AdversaryTrainingExample(symmetry3, Game.MAX_PLAYER, Nd4j.zeros(TicTacToeConstants.COLUMN_COUNT), 0)));
    assertTrue(uniqueSymmetries.contains(
        new AdversaryTrainingExample(symmetry4, Game.MAX_PLAYER, Nd4j.zeros(TicTacToeConstants.COLUMN_COUNT), 0)));
    assertTrue(uniqueSymmetries.contains(
        new AdversaryTrainingExample(symmetry5, Game.MAX_PLAYER, Nd4j.zeros(TicTacToeConstants.COLUMN_COUNT), 0)));
    assertTrue(uniqueSymmetries.contains(
        new AdversaryTrainingExample(symmetry6, Game.MAX_PLAYER, Nd4j.zeros(TicTacToeConstants.COLUMN_COUNT), 0)));
    assertTrue(uniqueSymmetries.contains(
        new AdversaryTrainingExample(symmetry7, Game.MAX_PLAYER, Nd4j.zeros(TicTacToeConstants.COLUMN_COUNT), 0)));
  }
}
