package ch.evolutionsoft.rl.alphazero.connectfour.playground;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

public class FieldTest {

	Field fieldC2, fieldD2, fieldE5, fieldE6;

	@BeforeEach
	public void setUp() {
		fieldC2 = new Field(21);
		fieldD2 = new Field(22);
		fieldE5 = new Field(50);
		fieldE6 = new Field(59);

	}
	

    @Test	
    public void testIsValidFieldString() {
        assertFalse(Field.isValidFieldString("a"));
        assertFalse(Field.isValidFieldString("a11"));   
        assertFalse(Field.isValidFieldString("11"));
        assertFalse(Field.isValidFieldString("\n5"));
        assertFalse(Field.isValidFieldString("a\n"));
        assertFalse(Field.isValidFieldString("h1"));
        assertFalse(Field.isValidFieldString("a7"));
        assertFalse(Field.isValidFieldString("a0"));
        
        assertTrue(Field.isValidFieldString("a1"));
        assertTrue(Field.isValidFieldString("a6"));
        assertTrue(Field.isValidFieldString("g1"));
        assertTrue(Field.isValidFieldString("g6"));
    
    }	

    @Test
    public void testEqualsOtherField() {
		assertFalse(this.fieldC2.equals(this.fieldD2));
        assertFalse(this.fieldE5.equals(this.fieldE6));
		
        assertEquals(fieldC2, fieldC2);
		assertEquals(fieldC2, new Field(21));
        assertEquals(fieldD2, new Field(22));
        assertEquals(fieldE5, new Field(50));
        assertEquals(fieldE6, new Field(59));
	}

    @Test
	public void testEqualsString() {
		assertEquals(fieldC2, "c2");
		assertEquals(fieldC2, "C2");
        assertEquals(fieldD2, "d2");
        assertEquals(fieldD2, "D2");
        assertEquals(fieldE5, "e5");
        assertEquals(fieldE5, "E5");
        assertEquals(fieldE6, "e6");
        assertEquals(fieldE6, "E6");
	}

    @Test
	public void testConstructorWithString() {
		
        Field fieldE3 = new Field("e3");
        assertEquals(32, fieldE3.getPosition());
	}

    @Test
	public void testConstructorWithInvalidStringArgument() {
		try {
			new Field("a21");
			fail("Should never reach this statement");
		} 
        catch (IllegalArgumentException IAE) {}
		try {
			new Field("h1");
			fail("Should never reach this statement");
		} 
        catch (IllegalArgumentException IAE) {}
		try {
			new Field("a7");
			fail("Should never reach this statement");
		} 
        catch (IllegalArgumentException IAE) {}
	}

	/*public void testConstructorWithArgumentsOutOfRange() {
		try {
			new Field(-1, 0);
			FieldTest.fail("Should never reach this statement");
		} 
        catch (IllegalArgumentException IAE) {}
		try {
			new Field(0, -1);
			FieldTest.fail("Should never reach this statement");
		} 
        catch (IllegalArgumentException IAE) {}
		try {
			new Field(6, 0);
			FieldTest.fail("Should never reach this statement");
		} 
        catch (IllegalArgumentException IAE) {}
		try {
			new Field(0, 7);
			FieldTest.fail("Should never reach this statement");
		} 
        catch (IllegalArgumentException IAE) {}
	}*/
}