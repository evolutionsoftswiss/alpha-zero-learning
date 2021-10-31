package ch.evolutionsoft.rl.alphazero.connectfour;

import static org.junit.jupiter.api.Assertions.*;

import java.util.List;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.factory.Nd4j;

import ch.evolutionsoft.rl.AdversaryTrainingExample;
import ch.evolutionsoft.rl.Game;
import ch.evolutionsoft.rl.alphazero.connectfour.playground.ArrayPlaygroundConstants;

class ConnectFourTest {

  ConnectFour connectFour = new ConnectFour();
  
  @Test
  void testCreateNewInstance() {
    
    ConnectFour newInstance = (ConnectFour) this.connectFour.createNewInstance();
    
    assertNotSame(this.connectFour.getCurrentBoard(), newInstance.getCurrentBoard());
    assertEquals(this.connectFour.getCurrentBoard(), newInstance.getCurrentBoard());
    
    assertNotSame(this.connectFour.arrayPlayground, newInstance.arrayPlayground);
    assertArrayEquals(
        this.connectFour.arrayPlayground.getColumnHeights(),
        newInstance.arrayPlayground.getColumnHeights());
    assertArrayEquals(
        this.connectFour.arrayPlayground.getPosition(),
        newInstance.arrayPlayground.getPosition());
    
  }
  
  @Test
  void testCreateNewInstanceWithStones() {

    this.connectFour.makeMove(0, Game.MAX_PLAYER);
    this.connectFour.makeMove(0, Game.MIN_PLAYER);
    
    ConnectFour newInstance = (ConnectFour) this.connectFour.createNewInstance();
    
    assertNotSame(this.connectFour.getCurrentBoard(), newInstance.getCurrentBoard());
    assertEquals(this.connectFour.getCurrentBoard(), newInstance.getCurrentBoard());
    
    assertNotSame(this.connectFour.arrayPlayground, newInstance.arrayPlayground);
    assertArrayEquals(
        this.connectFour.arrayPlayground.getColumnHeights(),
        newInstance.arrayPlayground.getColumnHeights());
    assertArrayEquals(
        this.connectFour.arrayPlayground.getPosition(),
        newInstance.arrayPlayground.getPosition());
    
  }
  
  @Test
  void testGetSymmetriesNoDifferentSymmetry() {

    this.connectFour.makeMove(3, Game.MAX_PLAYER);
    this.connectFour.makeMove(3, Game.MIN_PLAYER);
    
    AdversaryTrainingExample middleColumnTwoStonesExample =
        new AdversaryTrainingExample(
            this.connectFour.getCurrentBoard(),
            Game.MAX_PLAYER,
            Nd4j.zeros(ArrayPlaygroundConstants.COLUMN_COUNT),
            -1);
    
    List<AdversaryTrainingExample> symmetries = this.connectFour.getSymmetries(
        this.connectFour.getCurrentBoard(),
        Nd4j.zeros(ArrayPlaygroundConstants.COLUMN_COUNT),
        Game.MAX_PLAYER,
        -1);
    
    assertEquals(1, symmetries.size());
    assertEquals(middleColumnTwoStonesExample, symmetries.get(0));
  }
  
  @Test
  void testGetSymmetriesDifferentSymmetry() {

    this.connectFour.makeMove(3, Game.MAX_PLAYER);
    this.connectFour.makeMove(3, Game.MIN_PLAYER);
    Game newMirroredInstance = this.connectFour.createNewInstance();
    this.connectFour.makeMove(1, Game.MAX_PLAYER);
    this.connectFour.makeMove(4, Game.MIN_PLAYER);
    newMirroredInstance.makeMove(5, Game.MAX_PLAYER);
    newMirroredInstance.makeMove(2, Game.MIN_PLAYER);
    AdversaryTrainingExample fourStonesMirroredExample =
        new AdversaryTrainingExample(
            newMirroredInstance.getCurrentBoard(),
            Game.MAX_PLAYER,
            Nd4j.zeros(ArrayPlaygroundConstants.COLUMN_COUNT),
            -1);
    
    List<AdversaryTrainingExample> symmetries = this.connectFour.getSymmetries(
        this.connectFour.getCurrentBoard(),
        Nd4j.zeros(ArrayPlaygroundConstants.COLUMN_COUNT),
        Game.MAX_PLAYER,
        -1);
    
    assertEquals(1, symmetries.size());
    assertEquals(fourStonesMirroredExample, symmetries.get(0));
  }
}