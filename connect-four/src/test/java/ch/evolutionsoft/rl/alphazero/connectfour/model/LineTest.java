package ch.evolutionsoft.rl.alphazero.connectfour.model;

import static org.junit.jupiter.api.Assertions.assertSame;

import org.junit.jupiter.api.Test;

class LineTest {

  @Test
  void testVerticalBeginningFieldBottom() {

    Field beginningField = new Field("A1");
    Field endField = new Field("A4");

    Line line = new Line(beginningField, endField, PlaygroundConstants.RED);

    assertSame(4, line.getWinningRowLength());
    assertSame(0, line.getWinningRowColumnDirection());
    assertSame(1, line.getWinningRowRowDirection());
  }

  @Test
  void testVerticalBeginningFieldUp() {

    Field beginningField = new Field("A4");
    Field endField = new Field("A1");

    Line line = new Line(beginningField, endField, PlaygroundConstants.RED);

    assertSame(4, line.getWinningRowLength());
    assertSame(0, line.getWinningRowColumnDirection());
    assertSame(-1, line.getWinningRowRowDirection());
  }

  @Test
  void testHorizontalBeginningFieldLeft() {

    Field beginningField = new Field("C3");
    Field endField = new Field("G3");

    Line line = new Line(beginningField, endField, PlaygroundConstants.RED);

    assertSame(4, line.getWinningRowLength());
    assertSame(1, line.getWinningRowColumnDirection());
    assertSame(0, line.getWinningRowRowDirection());
  }

  @Test
  void testHorizontalBeginningFieldRight() {

    Field beginningField = new Field("G3");
    Field endField = new Field("C3");

    Line line = new Line(beginningField, endField, PlaygroundConstants.RED);

    assertSame(4, line.getWinningRowLength());
    assertSame(-1, line.getWinningRowColumnDirection());
    assertSame(0, line.getWinningRowRowDirection());
  }

  @Test
  void testDiagonallyDownBeginningFieldLeft() {

    Field beginningField = new Field("C6");
    Field endField = new Field("G3");

    Line line = new Line(beginningField, endField, PlaygroundConstants.RED);

    assertSame(4, line.getWinningRowLength());
    assertSame(1, line.getWinningRowColumnDirection());
    assertSame(-1, line.getWinningRowRowDirection());
  }

  @Test
  void testDiagonallyDownBeginningFieldRight() {

    Field beginningField = new Field("G3");
    Field endField = new Field("C6");

    Line line = new Line(beginningField, endField, PlaygroundConstants.RED);

    assertSame(4, line.getWinningRowLength());
    assertSame(-1, line.getWinningRowColumnDirection());
    assertSame(1, line.getWinningRowRowDirection());
  }

  @Test
  void testDiagonallyUpBeginningFieldLeft() {

    Field beginningField = new Field(31);
    Field endField = new Field(61);

    Line line = new Line(beginningField, endField, PlaygroundConstants.RED);

    assertSame(4, line.getWinningRowLength());
    assertSame(1, line.getWinningRowColumnDirection());
    assertSame(1, line.getWinningRowRowDirection());
  }

  @Test
  void testDiagonallyUpBeginningFieldRight() {

    Field beginningField = new Field(61);
    Field endField = new Field(31);

    Line line = new Line(beginningField, endField, PlaygroundConstants.RED);

    assertSame(4, line.getWinningRowLength());
    assertSame(-1, line.getWinningRowColumnDirection());
    assertSame(-1, line.getWinningRowRowDirection());
  }
}
