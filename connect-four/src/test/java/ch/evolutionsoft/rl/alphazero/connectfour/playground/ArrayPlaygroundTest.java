package ch.evolutionsoft.rl.alphazero.connectfour.playground;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class ArrayPlaygroundTest {

	protected ArrayPlayground arrayPlayground;

	@BeforeEach
	void setUp(){
		
		arrayPlayground = new ArrayPlayground();
	}
	
	@Test
	void testConstructor(){
		
		int[] playground = this.arrayPlayground.getPosition();
		
		for (int position = 0; position < 72; position++){
			
			if (position < 9 || position >= 63 ||
				position % 9 == 0 || (position + 1) % 9 == 0){
				
				assertSame(ArrayPlaygroundConstants.GREY, playground[position]);
			}		
			else{
				
				assertSame(ArrayPlaygroundConstants.EMPTY, playground[position]);
			}
		}
	}
	

    @Test
	void testTrySetFieldColorAndPositionEmptyColumn(){
		
		this.arrayPlayground.trySetField(0, ArrayPlaygroundConstants.YELLOW);
		
		int[] playground = this.arrayPlayground.getPosition();
		
		assertSame(ArrayPlaygroundConstants.YELLOW, playground[10]);
	}
	

    @Test
	void testTrySetFieldColumnHeight(){
		
		this.arrayPlayground.trySetField(0, ArrayPlaygroundConstants.YELLOW);
		
		int[] columnHeights = this.arrayPlayground.getColumnHeights();
		
		assertSame(1, columnHeights[0]);
	}
	

    @Test
	void testTrySetFieldColorAndPositionNonEmptyColumn(){
		
		this.arrayPlayground.trySetField(0, ArrayPlaygroundConstants.YELLOW);
		this.arrayPlayground.trySetField(0, ArrayPlaygroundConstants.YELLOW);
		
		int[] playground = this.arrayPlayground.getPosition();
		
		assertSame(ArrayPlaygroundConstants.YELLOW, playground[19]);
	}
	

    @Test
	void testTrySetFieldEmpty(){
		
		this.arrayPlayground.trySetField(0, ArrayPlaygroundConstants.YELLOW);
		this.arrayPlayground.trySetField(1, ArrayPlaygroundConstants.RED);
		
		this.arrayPlayground.trySetFieldEmpty(1);
		
		assertSame(41, this.arrayPlayground.getFieldsLeft());
		
		int[] playground = this.arrayPlayground.getPosition();
		
		assertSame(ArrayPlaygroundConstants.EMPTY, playground[11]);
	}
	

    @Test
	void testFourInARowHorizontallyNot4Stones(){
		
		this.arrayPlayground.trySetField(2, ArrayPlaygroundConstants.RED);
		this.arrayPlayground.trySetField(3, ArrayPlaygroundConstants.RED);
		this.arrayPlayground.trySetField(4, ArrayPlaygroundConstants.RED);
		this.arrayPlayground.trySetField(5, ArrayPlaygroundConstants.RED);
		this.arrayPlayground.trySetField(2, ArrayPlaygroundConstants.RED);
		this.arrayPlayground.trySetField(3, ArrayPlaygroundConstants.YELLOW);
		this.arrayPlayground.trySetField(4, ArrayPlaygroundConstants.YELLOW);
		this.arrayPlayground.trySetField(5, ArrayPlaygroundConstants.YELLOW);
		
		assertFalse(this.arrayPlayground.fourInARowHorizontally(this.arrayPlayground.getLastPosition(5), ArrayPlaygroundConstants.YELLOW));
	}	
	

    @Test
	void testFourInARowHorizontally4Stones(){
		
		this.arrayPlayground.trySetField(2, ArrayPlaygroundConstants.RED);
		this.arrayPlayground.trySetField(3, ArrayPlaygroundConstants.RED);
		this.arrayPlayground.trySetField(4, ArrayPlaygroundConstants.RED);
		this.arrayPlayground.trySetField(5, ArrayPlaygroundConstants.RED);
		this.arrayPlayground.trySetField(2, ArrayPlaygroundConstants.YELLOW);
		this.arrayPlayground.trySetField(3, ArrayPlaygroundConstants.YELLOW);
		this.arrayPlayground.trySetField(4, ArrayPlaygroundConstants.YELLOW);
		this.arrayPlayground.trySetField(5, ArrayPlaygroundConstants.YELLOW);
		
		assertTrue(this.arrayPlayground.fourInARowHorizontally(this.arrayPlayground.getLastPosition(5), ArrayPlaygroundConstants.YELLOW));
	}
	

    @Test
	void testFourInARowVerticallyNot4Stones(){
		
		this.arrayPlayground.trySetField(2, ArrayPlaygroundConstants.RED);
		this.arrayPlayground.trySetField(2, ArrayPlaygroundConstants.RED);
		this.arrayPlayground.trySetField(2, ArrayPlaygroundConstants.RED);
		this.arrayPlayground.trySetField(2, ArrayPlaygroundConstants.YELLOW);
		
		assertFalse(this.arrayPlayground.fourInARowVertically(this.arrayPlayground.getLastPosition(2), ArrayPlaygroundConstants.RED));
	}	
	

    @Test
	void testFourInARowVertically4Stones(){
		
		this.arrayPlayground.trySetField(2, ArrayPlaygroundConstants.RED);
		this.arrayPlayground.trySetField(2, ArrayPlaygroundConstants.RED);
		this.arrayPlayground.trySetField(2, ArrayPlaygroundConstants.RED);
		this.arrayPlayground.trySetField(2, ArrayPlaygroundConstants.RED);
		
		assertTrue(this.arrayPlayground.fourInARowVertically(this.arrayPlayground.getLastPosition(2), ArrayPlaygroundConstants.RED));
	}
	

    @Test
	void testFourInARowDiagonallyDownNot4Stones(){

		
		this.arrayPlayground.trySetField(2, ArrayPlaygroundConstants.RED);
		this.arrayPlayground.trySetField(3, ArrayPlaygroundConstants.RED);
		this.arrayPlayground.trySetField(4, ArrayPlaygroundConstants.RED);
		this.arrayPlayground.trySetField(5, ArrayPlaygroundConstants.YELLOW);
		
		this.arrayPlayground.trySetField(2, ArrayPlaygroundConstants.RED);
		this.arrayPlayground.trySetField(3, ArrayPlaygroundConstants.RED);
		this.arrayPlayground.trySetField(4, ArrayPlaygroundConstants.YELLOW);
		
		this.arrayPlayground.trySetField(2, ArrayPlaygroundConstants.RED);
		this.arrayPlayground.trySetField(3, ArrayPlaygroundConstants.YELLOW);
		
		assertFalse(this.arrayPlayground.fourInARowDiagonallyDown(this.arrayPlayground.getLastPosition(3), ArrayPlaygroundConstants.YELLOW));
	}	
	

    @Test
	void testFourInARowDiagonallyDown4Stones(){
		
		this.arrayPlayground.trySetField(2, ArrayPlaygroundConstants.RED);
		this.arrayPlayground.trySetField(3, ArrayPlaygroundConstants.RED);
		this.arrayPlayground.trySetField(4, ArrayPlaygroundConstants.RED);
		this.arrayPlayground.trySetField(5, ArrayPlaygroundConstants.YELLOW);
		
		this.arrayPlayground.trySetField(2, ArrayPlaygroundConstants.RED);
		this.arrayPlayground.trySetField(3, ArrayPlaygroundConstants.RED);
		this.arrayPlayground.trySetField(4, ArrayPlaygroundConstants.YELLOW);
		
		this.arrayPlayground.trySetField(2, ArrayPlaygroundConstants.RED);
		this.arrayPlayground.trySetField(3, ArrayPlaygroundConstants.YELLOW);
		
		this.arrayPlayground.trySetField(2, ArrayPlaygroundConstants.YELLOW);
		
		assertTrue(this.arrayPlayground.fourInARowDiagonallyDown(this.arrayPlayground.getLastPosition(2), ArrayPlaygroundConstants.YELLOW));
	}
	

    @Test
	void testFourInARowDiagonallyUpNot4Stones(){

		
		this.arrayPlayground.trySetField(2, ArrayPlaygroundConstants.YELLOW);
		this.arrayPlayground.trySetField(3, ArrayPlaygroundConstants.RED);
		this.arrayPlayground.trySetField(4, ArrayPlaygroundConstants.RED);
		this.arrayPlayground.trySetField(5, ArrayPlaygroundConstants.RED);
		
		this.arrayPlayground.trySetField(3, ArrayPlaygroundConstants.YELLOW);
		this.arrayPlayground.trySetField(4, ArrayPlaygroundConstants.RED);
		this.arrayPlayground.trySetField(5, ArrayPlaygroundConstants.RED);
		
		this.arrayPlayground.trySetField(4, ArrayPlaygroundConstants.YELLOW);
		this.arrayPlayground.trySetField(5, ArrayPlaygroundConstants.RED);
		
		assertFalse(this.arrayPlayground.fourInARowDiagonallyUp(this.arrayPlayground.getLastPosition(3), ArrayPlaygroundConstants.YELLOW));
	}	
	

    @Test
	void testFourInARowDiagonallyUp4Stones(){
		
		this.arrayPlayground.trySetField(2, ArrayPlaygroundConstants.YELLOW);
		this.arrayPlayground.trySetField(3, ArrayPlaygroundConstants.RED);
		this.arrayPlayground.trySetField(4, ArrayPlaygroundConstants.RED);
		this.arrayPlayground.trySetField(5, ArrayPlaygroundConstants.RED);
		
		this.arrayPlayground.trySetField(3, ArrayPlaygroundConstants.YELLOW);
		this.arrayPlayground.trySetField(4, ArrayPlaygroundConstants.RED);
		this.arrayPlayground.trySetField(5, ArrayPlaygroundConstants.RED);
		
		this.arrayPlayground.trySetField(4, ArrayPlaygroundConstants.YELLOW);
		this.arrayPlayground.trySetField(5, ArrayPlaygroundConstants.RED);
		
		this.arrayPlayground.trySetField(5, ArrayPlaygroundConstants.YELLOW);
		
		assertTrue(this.arrayPlayground.fourInARowDiagonallyUp(this.arrayPlayground.getLastPosition(5), ArrayPlaygroundConstants.YELLOW));
	}
}
