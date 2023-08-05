package ch.evolutionsoft.rl.alphazero.connectfour.model;

/**
 * @author evolutionsoft
 */
public class Field {

  protected int position;

  public Field(int position) {

    this.position = position;
  }

  public Field(String fieldString) {

    if (!isValidFieldString(fieldString)) {
      throw new IllegalArgumentException("Invalid String for a field.");
    }

    char columnChar = fieldString.charAt(0);
    char rowChar = fieldString.charAt(1);

    if (Character.isLowerCase(columnChar)) {

      this.position = (rowChar - '1') * PlaygroundConstants.COLUMN_COUNT + (columnChar - 'a');
    } else {

      this.position = (rowChar - '1') * PlaygroundConstants.COLUMN_COUNT + (columnChar - 'A');
    }
  }

  public int getPosition() {

    return this.position;
  }

  public int getColumn() {

    return this.position % PlaygroundConstants.COLUMN_COUNT;
  }

  public int getRow() {

    return this.position / PlaygroundConstants.COLUMN_COUNT;
  }

  public boolean equalsField(Field otherField) {

    return this.position == otherField.position;
  }

  public boolean equalsString(String fieldString) {

    if (fieldString.length() == 2) {

      char columnChar = fieldString.charAt(0);
      char rowChar = fieldString.charAt(1);

      if (Character.isLowerCase(columnChar)) {

        return this.position / PlaygroundConstants.COLUMN_COUNT == (rowChar - '1') &&
            this.position % PlaygroundConstants.COLUMN_COUNT == (columnChar - 'a');
      }

      if (Character.isUpperCase(columnChar)) {

        return this.position / PlaygroundConstants.COLUMN_COUNT == (rowChar - '1') &&
            this.position % PlaygroundConstants.COLUMN_COUNT == (columnChar - 'A');
      }
    }
    return false;
  }

  protected static boolean isValidFieldString(String fieldString) {
    if (fieldString.length() == 2) {
      char firstChar = fieldString.charAt(0);
      char secondChar = fieldString.charAt(1);
      if (Character.isLowerCase(firstChar))
        return firstChar >= 'a' && firstChar <= 'g'
            && secondChar >= '1' && secondChar <= '6';
      return firstChar >= 'A' && firstChar <= 'G'
          && secondChar >= '1' && secondChar <= '6';
    }
    return false;
  }
}
