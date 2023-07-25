package ch.evolutionsoft.rl.alphazero.connectfour.playground;

import static ch.evolutionsoft.rl.alphazero.connectfour.playground.PlaygroundConstants.*;
import static org.junit.jupiter.api.Assertions.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class BinaryPlaygroundTest {

  private BinaryPlayground binaryPlayGround, emptyPlayGround;
  private int[] testPosition;
  private int[] columnHeights;

  private int fieldsLeftInTestPosition;

  private ArrayPosition arrayPosition;

  @BeforeEach
  protected void setUp() {

    binaryPlayGround = new BinaryPlayground();
    emptyPlayGround = new BinaryPlayground();

    testPosition = new int[] { 3, 3, 3, 3, 3, 3, 3, 3, 3,
        3, 0, 1, 2, 0, 2, 2, 2, 3,
        3, 1, 0, 2, 1, 2, 2, 2, 3,
        3, 2, 2, 2, 2, 2, 2, 2, 3,
        3, 2, 2, 2, 2, 2, 2, 2, 3,
        3, 2, 2, 2, 2, 2, 2, 2, 3,
        3, 2, 2, 2, 2, 2, 2, 2, 3,
        3, 3, 3, 3, 3, 3, 3, 3, 3 };

    columnHeights = new int[] { 2, 2, 0, 2, 0, 0, 0 };

    fieldsLeftInTestPosition = 36;

    arrayPosition = new ArrayPosition(testPosition, columnHeights);

    this.binaryPlayGround = new BinaryPlayground(arrayPosition, fieldsLeftInTestPosition);
    this.emptyPlayGround = new BinaryPlayground();
  }

  @Test
  void testConstructorArgBinaryPlayGround() {

    BinaryPlayground copy = new BinaryPlayground(this.binaryPlayGround);

    assertEquals(this.binaryPlayGround.getFieldsLeft(), copy.getFieldsLeft());
    assertNotSame(this.binaryPlayGround.getPosition(), copy.getPosition());
    assertNotSame(this.binaryPlayGround.getFirstFreeBitOfColumn(), copy.getFirstFreeBitOfColumn());
    assertTrue(Arrays.equals((long[]) this.binaryPlayGround.getPosition(), (long[]) copy.getPosition()));
    assertTrue(Arrays.equals(this.binaryPlayGround.getFirstFreeBitOfColumn(), copy.getFirstFreeBitOfColumn()));
  }

  @Test
  void testMakePosition() {

    long[] playGroundColors = this.emptyPlayGround.makePosition(this.testPosition);

    long expectedFirstPlayerColor = 2097409L;
    long expectedSecondPlayerColor = 4194434L;

    assertEquals(expectedFirstPlayerColor, playGroundColors[0]);
    assertEquals(expectedSecondPlayerColor, playGroundColors[1]);
  }

  @Test
  void testMakeFirstFreeBitOfColumn() {

    int[] heightOfColumn = new int[] { 3, 4, 0, 1, 0, 0, 6 };
    int[] firstFreeBitOfColumn = this.emptyPlayGround.makeFirstFreeBitOfColumn(heightOfColumn);

    assertEquals(3, firstFreeBitOfColumn[0]);
    assertEquals(11, firstFreeBitOfColumn[1]);
    assertEquals(14, firstFreeBitOfColumn[2]);
    assertEquals(22, firstFreeBitOfColumn[3]);
    assertEquals(28, firstFreeBitOfColumn[4]);
    assertEquals(35, firstFreeBitOfColumn[5]);
    assertEquals(48, firstFreeBitOfColumn[6]);
  }

  @Test
  void testEnterMove() {

    this.binaryPlayGround.trySetField(1, YELLOW);
    assertEquals(this.fieldsLeftInTestPosition - 1, this.binaryPlayGround.getFieldsLeft());
    assertEquals(3, this.binaryPlayGround.getHeightOfColumn(1));
    assertEquals(10, this.binaryPlayGround.getFirstFreeBitOfColumn(1));
    assertEquals(4194434L, this.binaryPlayGround.getSecondPlayerPosition());
    assertEquals(2097921L, this.binaryPlayGround.getFirstPlayerPosition());

    this.binaryPlayGround.trySetField(1, RED);

    assertEquals(this.fieldsLeftInTestPosition - 2, this.binaryPlayGround.getFieldsLeft());
    assertEquals(4, this.binaryPlayGround.getHeightOfColumn(1));
    assertEquals(11, this.binaryPlayGround.getFirstFreeBitOfColumn(1));
    assertEquals(2097921L, this.binaryPlayGround.getFirstPlayerPosition());
    assertEquals(4195458L, this.binaryPlayGround.getSecondPlayerPosition());
  }

  @Test
  void testTakeBackMove() {

    this.binaryPlayGround.trySetField(1, YELLOW);
    this.binaryPlayGround.trySetField(1, RED);
    this.binaryPlayGround.trySetFieldEmpty(1, RED);

    assertEquals(this.fieldsLeftInTestPosition - 1, this.binaryPlayGround.getFieldsLeft());
    assertEquals(3, this.binaryPlayGround.getHeightOfColumn(1));
    assertEquals(10, this.binaryPlayGround.getFirstFreeBitOfColumn(1));
    assertEquals(4194434L, this.binaryPlayGround.getSecondPlayerPosition());
    assertEquals(2097921L, this.binaryPlayGround.getFirstPlayerPosition());

    this.binaryPlayGround.trySetFieldEmpty(1, YELLOW);

    assertEquals(this.fieldsLeftInTestPosition, this.binaryPlayGround.getFieldsLeft());
    assertEquals(2, this.binaryPlayGround.getHeightOfColumn(1));
    assertEquals(9, this.binaryPlayGround.getFirstFreeBitOfColumn(1));
    assertEquals(4194434L, this.binaryPlayGround.getSecondPlayerPosition());
    assertEquals(2097409L, this.binaryPlayGround.getFirstPlayerPosition());
  }

  @Test
  void testFourInARowHorizontal() {

    int[] position = new int[72];

    position[10] = RED;
    position[11] = RED;
    position[12] = RED;
    position[13] = RED;

    ArrayPosition arrayPosition = new ArrayPosition(position, new int[] { 1, 1, 1, 1, -1, -1, -1 });

    BinaryPlayground playGround = new BinaryPlayground(arrayPosition, 38);
    assertTrue(playGround.fourInARow(playGround.getSecondPlayerPosition()));
  }

  @Test
  void testFourInARowVertical() {

    int[] position = new int[72];

    position[15] = RED;
    position[24] = RED;
    position[33] = RED;
    position[42] = RED;

    ArrayPosition arrayPosition = new ArrayPosition(position, new int[] { -1, -1, -1, -1, -1, -1, -1 });

    BinaryPlayground playGround = new BinaryPlayground(arrayPosition, 38);
    assertTrue(playGround.fourInARow(playGround.getSecondPlayerPosition()));
  }

  @Test
  void testFourInARowDiagonalNorthSouth() {

    int[] position = new int[72];

    position[38] = RED;
    position[30] = RED;
    position[22] = RED;
    position[14] = RED;

    ArrayPosition arrayPosition = new ArrayPosition(position, new int[] { -1, -1, -1, -1, -1, -1, -1 });

    BinaryPlayground playGround = new BinaryPlayground(arrayPosition, 38);
    assertTrue(playGround.fourInARow(playGround.getSecondPlayerPosition()));
  }

  @Test
  void testFourInARowDiagonalSouthNorth() {

    int[] position = new int[72];

    position[11] = RED;
    position[21] = RED;
    position[31] = RED;
    position[41] = RED;

    ArrayPosition arrayPosition = new ArrayPosition(position, new int[] { -1, -1, -1, -1, -1, -1, -1 });

    BinaryPlayground playGround = new BinaryPlayground(arrayPosition, 38);
    assertTrue(playGround.fourInARow(playGround.getSecondPlayerPosition()));
  }

  @Test
  void testPlayableColumnsPriorityOrderedColumnFirst() {

    ArrayPosition arrayPosition = new ArrayPosition(this.testPosition, this.columnHeights);

    this.binaryPlayGround = new BinaryPlayground(arrayPosition, 36);
    List<Integer> expectedPlayableColumnsPriorityOrdered = new ArrayList<Integer>();
    expectedPlayableColumnsPriorityOrdered.add(3);
    expectedPlayableColumnsPriorityOrdered.add(2);
    expectedPlayableColumnsPriorityOrdered.add(4);
    expectedPlayableColumnsPriorityOrdered.add(1);
    expectedPlayableColumnsPriorityOrdered.add(5);
    expectedPlayableColumnsPriorityOrdered.add(0);
    expectedPlayableColumnsPriorityOrdered.add(6);
    List<Integer> playableColumnsPriorityOrdered = this.binaryPlayGround.playableColumnsPriorityOrderedColumnFirst(3);

    assertEquals(expectedPlayableColumnsPriorityOrdered, playableColumnsPriorityOrdered);

    expectedPlayableColumnsPriorityOrdered = new ArrayList<Integer>();
    expectedPlayableColumnsPriorityOrdered.add(6);
    expectedPlayableColumnsPriorityOrdered.add(3);
    expectedPlayableColumnsPriorityOrdered.add(2);
    expectedPlayableColumnsPriorityOrdered.add(4);
    expectedPlayableColumnsPriorityOrdered.add(1);
    expectedPlayableColumnsPriorityOrdered.add(5);
    expectedPlayableColumnsPriorityOrdered.add(0);
    playableColumnsPriorityOrdered = this.binaryPlayGround.playableColumnsPriorityOrderedColumnFirst(6);

    assertEquals(expectedPlayableColumnsPriorityOrdered, playableColumnsPriorityOrdered);

    expectedPlayableColumnsPriorityOrdered = new ArrayList<Integer>();
    expectedPlayableColumnsPriorityOrdered.add(1);
    expectedPlayableColumnsPriorityOrdered.add(3);
    expectedPlayableColumnsPriorityOrdered.add(2);
    expectedPlayableColumnsPriorityOrdered.add(4);
    expectedPlayableColumnsPriorityOrdered.add(5);
    expectedPlayableColumnsPriorityOrdered.add(0);
    expectedPlayableColumnsPriorityOrdered.add(6);
    playableColumnsPriorityOrdered = this.binaryPlayGround.playableColumnsPriorityOrderedColumnFirst(1);

    assertEquals(expectedPlayableColumnsPriorityOrdered, playableColumnsPriorityOrdered);

  }
}
