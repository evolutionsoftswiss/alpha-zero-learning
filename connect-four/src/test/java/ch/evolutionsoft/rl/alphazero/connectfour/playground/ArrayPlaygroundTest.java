package ch.evolutionsoft.rl.alphazero.connectfour.playground;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

public class ArrayPlaygroundTest {

	protected ArrayPlayground arrayPlayground;

	@BeforeEach
	public void setUp(){
		
		arrayPlayground = new ArrayPlayground();
	}
	
	@Test
	public void testConstructor(){
		
		int[] playground = this.arrayPlayground.getPosition();
		
		for (int position = 0; position < 72; position++){
			
			if (position < 9 || position >= 63 ||
				position % 9 == 0 || (position + 1) % 9 == 0){
				
				assertSame(Playground.GREY, playground[position]);
			}		
			else{
				
				assertSame(Playground.EMPTY, playground[position]);
			}
		}
	}
	

    @Test
	public void testTrySetFieldColorAndPositionEmptyColumn(){
		
		this.arrayPlayground.trySetField(0, Playground.YELLOW);
		
		int[] playground = this.arrayPlayground.getPosition();
		
		assertSame(Playground.YELLOW, playground[10]);
	}
	

    @Test
	public void testTrySetFieldColumnHeight(){
		
		this.arrayPlayground.trySetField(0, Playground.YELLOW);
		
		int[] columnHeights = this.arrayPlayground.getColumnHeights();
		
		assertSame(1, columnHeights[0]);
	}
	

    @Test
	public void testTrySetFieldColorAndPositionNonEmptyColumn(){
		
		this.arrayPlayground.trySetField(0, Playground.YELLOW);
		this.arrayPlayground.trySetField(0, Playground.YELLOW);
		
		int[] playground = this.arrayPlayground.getPosition();
		
		assertSame(Playground.YELLOW, playground[19]);
	}
	

    @Test
	public void testTrySetFieldEmpty(){
		
		this.arrayPlayground.trySetField(0, Playground.YELLOW);
		this.arrayPlayground.trySetField(1, Playground.RED);
		
		this.arrayPlayground.trySetFieldEmpty(1);
		
		assertSame(41, this.arrayPlayground.getFieldsLeft());
		
		int[] playground = this.arrayPlayground.getPosition();
		
		assertSame(Playground.EMPTY, playground[11]);
	}
	

    @Test
	public void testFourInARowHorizontallyNot4Stones(){
		
		this.arrayPlayground.trySetField(2, Playground.RED);
		this.arrayPlayground.trySetField(3, Playground.RED);
		this.arrayPlayground.trySetField(4, Playground.RED);
		this.arrayPlayground.trySetField(5, Playground.RED);
		this.arrayPlayground.trySetField(2, Playground.RED);
		this.arrayPlayground.trySetField(3, Playground.YELLOW);
		this.arrayPlayground.trySetField(4, Playground.YELLOW);
		this.arrayPlayground.trySetField(5, Playground.YELLOW);
		
		assertFalse(this.arrayPlayground.fourInARowHorizontally(this.arrayPlayground.getLastPosition(5), Playground.YELLOW));
	}	
	

    @Test
	public void testFourInARowHorizontally4Stones(){
		
		this.arrayPlayground.trySetField(2, Playground.RED);
		this.arrayPlayground.trySetField(3, Playground.RED);
		this.arrayPlayground.trySetField(4, Playground.RED);
		this.arrayPlayground.trySetField(5, Playground.RED);
		this.arrayPlayground.trySetField(2, Playground.YELLOW);
		this.arrayPlayground.trySetField(3, Playground.YELLOW);
		this.arrayPlayground.trySetField(4, Playground.YELLOW);
		this.arrayPlayground.trySetField(5, Playground.YELLOW);
		
		assertTrue(this.arrayPlayground.fourInARowHorizontally(this.arrayPlayground.getLastPosition(5), Playground.YELLOW));
	}
	

    @Test
	public void testFourInARowVerticallyNot4Stones(){
		
		this.arrayPlayground.trySetField(2, Playground.RED);
		this.arrayPlayground.trySetField(2, Playground.RED);
		this.arrayPlayground.trySetField(2, Playground.RED);
		this.arrayPlayground.trySetField(2, Playground.YELLOW);
		
		assertFalse(this.arrayPlayground.fourInARowVertically(this.arrayPlayground.getLastPosition(2), Playground.RED));
	}	
	

    @Test
	public void testFourInARowVertically4Stones(){
		
		this.arrayPlayground.trySetField(2, Playground.RED);
		this.arrayPlayground.trySetField(2, Playground.RED);
		this.arrayPlayground.trySetField(2, Playground.RED);
		this.arrayPlayground.trySetField(2, Playground.RED);
		
		assertTrue(this.arrayPlayground.fourInARowVertically(this.arrayPlayground.getLastPosition(2), Playground.RED));
	}
	

    @Test
	public void testFourInARowDiagonallyDownNot4Stones(){

		
		this.arrayPlayground.trySetField(2, Playground.RED);
		this.arrayPlayground.trySetField(3, Playground.RED);
		this.arrayPlayground.trySetField(4, Playground.RED);
		this.arrayPlayground.trySetField(5, Playground.YELLOW);
		
		this.arrayPlayground.trySetField(2, Playground.RED);
		this.arrayPlayground.trySetField(3, Playground.RED);
		this.arrayPlayground.trySetField(4, Playground.YELLOW);
		
		this.arrayPlayground.trySetField(2, Playground.RED);
		this.arrayPlayground.trySetField(3, Playground.YELLOW);
		
		assertFalse(this.arrayPlayground.fourInARowDiagonallyDown(this.arrayPlayground.getLastPosition(3), Playground.YELLOW));
	}	
	

    @Test
	public void testFourInARowDiagonallyDown4Stones(){
		
		this.arrayPlayground.trySetField(2, Playground.RED);
		this.arrayPlayground.trySetField(3, Playground.RED);
		this.arrayPlayground.trySetField(4, Playground.RED);
		this.arrayPlayground.trySetField(5, Playground.YELLOW);
		
		this.arrayPlayground.trySetField(2, Playground.RED);
		this.arrayPlayground.trySetField(3, Playground.RED);
		this.arrayPlayground.trySetField(4, Playground.YELLOW);
		
		this.arrayPlayground.trySetField(2, Playground.RED);
		this.arrayPlayground.trySetField(3, Playground.YELLOW);
		
		this.arrayPlayground.trySetField(2, Playground.YELLOW);
		
		assertTrue(this.arrayPlayground.fourInARowDiagonallyDown(this.arrayPlayground.getLastPosition(2), Playground.YELLOW));
	}
	

    @Test
	public void testFourInARowDiagonallyUpNot4Stones(){

		
		this.arrayPlayground.trySetField(2, Playground.YELLOW);
		this.arrayPlayground.trySetField(3, Playground.RED);
		this.arrayPlayground.trySetField(4, Playground.RED);
		this.arrayPlayground.trySetField(5, Playground.RED);
		
		this.arrayPlayground.trySetField(3, Playground.YELLOW);
		this.arrayPlayground.trySetField(4, Playground.RED);
		this.arrayPlayground.trySetField(5, Playground.RED);
		
		this.arrayPlayground.trySetField(4, Playground.YELLOW);
		this.arrayPlayground.trySetField(5, Playground.RED);
		
		assertFalse(this.arrayPlayground.fourInARowDiagonallyUp(this.arrayPlayground.getLastPosition(3), Playground.YELLOW));
	}	
	

    @Test
	public void testFourInARowDiagonallyUp4Stones(){
		
		this.arrayPlayground.trySetField(2, Playground.YELLOW);
		this.arrayPlayground.trySetField(3, Playground.RED);
		this.arrayPlayground.trySetField(4, Playground.RED);
		this.arrayPlayground.trySetField(5, Playground.RED);
		
		this.arrayPlayground.trySetField(3, Playground.YELLOW);
		this.arrayPlayground.trySetField(4, Playground.RED);
		this.arrayPlayground.trySetField(5, Playground.RED);
		
		this.arrayPlayground.trySetField(4, Playground.YELLOW);
		this.arrayPlayground.trySetField(5, Playground.RED);
		
		this.arrayPlayground.trySetField(5, Playground.YELLOW);
		
		assertTrue(this.arrayPlayground.fourInARowDiagonallyUp(this.arrayPlayground.getLastPosition(5), Playground.YELLOW));
	}
}
