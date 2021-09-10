package ch.evolutionsoft.rl.alphazero.connectfour.playground;

import static org.junit.jupiter.api.Assertions.assertSame;

import org.junit.jupiter.api.Test;

public class LineTest {

	@Test
	public void testVerticalBeginningFieldBottom(){
		
		Field beginningField = new Field(10);
		Field endField = new Field(37);
		
		Line line = new Line(beginningField, endField, ArrayPlaygroundConstants.RED);
		
		assertSame(4, line.getWinningRowLength());
		assertSame(0, line.getWinningRowColumnDirection());
		assertSame(1, line.getWinningRowRowDirection());
	}
	

    @Test
	public void testVerticalBeginningFieldUp(){
		
		Field beginningField = new Field(37);
		Field endField = new Field(10);
		
		Line line = new Line(beginningField, endField, ArrayPlaygroundConstants.RED);
		
		assertSame(4, line.getWinningRowLength());
		assertSame(0, line.getWinningRowColumnDirection());
		assertSame(-1, line.getWinningRowRowDirection());
	}
	

    @Test
	public void testHorizontalBeginningFieldLeft(){
		
		Field beginningField = new Field(31);
		Field endField = new Field(34);
		
		Line line = new Line(beginningField, endField, ArrayPlaygroundConstants.RED);
		
		assertSame(4, line.getWinningRowLength());
		assertSame(1, line.getWinningRowColumnDirection());
		assertSame(0, line.getWinningRowRowDirection());
	}
	

    @Test
	public void testHorizontalBeginningFieldRight(){
		
		Field beginningField = new Field(34);
		Field endField = new Field(31);
		
		Line line = new Line(beginningField, endField, ArrayPlaygroundConstants.RED);
		
		assertSame(4, line.getWinningRowLength());
		assertSame(-1, line.getWinningRowColumnDirection());
		assertSame(0, line.getWinningRowRowDirection());
	}
	

    @Test
	public void testDiagonallyDownBeginningFieldLeft(){

		Field beginningField = new Field(55);
		Field endField = new Field(31);
		
		Line line = new Line(beginningField, endField, ArrayPlaygroundConstants.RED);
		
		assertSame(4, line.getWinningRowLength());
		assertSame(1, line.getWinningRowColumnDirection());
		assertSame(-1, line.getWinningRowRowDirection());
	}
	

    @Test
	public void testDiagonallyDownBeginningFieldRight(){

		Field beginningField = new Field(31);
		Field endField = new Field(55);
		
		Line line = new Line(beginningField, endField, ArrayPlaygroundConstants.RED);
		
		assertSame(4, line.getWinningRowLength());
		assertSame(-1, line.getWinningRowColumnDirection());
		assertSame(1, line.getWinningRowRowDirection());
	}
	

    @Test
	public void testDiagonallyUpBeginningFieldLeft(){

		Field beginningField = new Field(31);
		Field endField = new Field(61);
		
		Line line = new Line(beginningField, endField, ArrayPlaygroundConstants.RED);
		
		assertSame(4, line.getWinningRowLength());
		assertSame(1, line.getWinningRowColumnDirection());
		assertSame(1, line.getWinningRowRowDirection());
	}
	

    @Test
	public void testDiagonallyUpBeginningFieldRight(){

		Field beginningField = new Field(61);
		Field endField = new Field(31);
		
		Line line = new Line(beginningField, endField, ArrayPlaygroundConstants.RED);
		
		assertSame(4, line.getWinningRowLength());
		assertSame(-1, line.getWinningRowColumnDirection());
		assertSame(-1, line.getWinningRowRowDirection());
	}
}