package ch.evolutionsoft.rl.alphazero.connectfour.model;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class FieldTest {

  Field fieldC2;
  Field fieldD2;
  Field fieldE5;
  Field fieldE6;
  

  @BeforeEach
  void setUp() {
    fieldC2 = new Field(7 + 2);
    fieldD2 = new Field(7 + 3);
    fieldE5 = new Field(4 * 7 + 4);
    fieldE6 = new Field(5 * 7 + 4);

  }

  @Test
  void testIsValidFieldString() {
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
  void testEqualsString() {
    assertTrue(fieldC2.equalsString("c2"));
    assertTrue(fieldC2.equalsString("C2"));
    assertTrue(fieldD2.equalsString("d2"));
    assertTrue(fieldD2.equalsString("D2"));
    assertTrue(fieldE5.equalsString("e5"));
    assertTrue(fieldE5.equalsString("E5"));
    assertTrue(fieldE6.equalsString("e6"));
    assertTrue(fieldE6.equalsString("E6"));
  }

  @Test
  void testConstructorWithString() {

    Field fieldE3 = new Field("e3");
    assertEquals(2 * 7 + 4, fieldE3.getPosition());
  }

  @Test
  void testConstructorWithInvalidStringArgument() {
    try {
      new Field("a21");
      fail("Should never reach this statement");
    } catch (IllegalArgumentException IAE) {
    }
    try {
      new Field("h1");
      fail("Should never reach this statement");
    } catch (IllegalArgumentException IAE) {
    }
    try {
      new Field("a7");
      fail("Should never reach this statement");
    } catch (IllegalArgumentException IAE) {
    }
  }
}