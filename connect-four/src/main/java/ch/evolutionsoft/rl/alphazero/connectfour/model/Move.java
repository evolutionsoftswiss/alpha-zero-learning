package ch.evolutionsoft.rl.alphazero.connectfour.model;

public class Move extends Field {

  protected final int color;

  public Move(int index, int color) {
    super(index);
    this.color = color;
  }

  public Move(String moveString) {
    super(moveString);
    if (Character.isUpperCase(moveString.charAt(0)))
      this.color = PlaygroundConstants.YELLOW;
    else
      this.color = PlaygroundConstants.RED;
  }

  public int getColor() {
    return color;
  }

  public boolean equalsMove(Move move) {

    return this.color == move.color &&
        this.position == move.position;
  }

  @Override
  public boolean equalsString(String moveString) {

    if (moveString.length() == 2) {
      char firstChar = moveString.charAt(0);
      return super.equalsString(moveString)
          && (Character.isLowerCase(firstChar) && this.color == PlaygroundConstants.RED
              || Character.isUpperCase(firstChar) && this.color == PlaygroundConstants.YELLOW);
    }
    return false;
  }

  public String toString() {

    byte[] columnAsciiValues;
    if (this.color == PlaygroundConstants.RED)
      columnAsciiValues = new byte[] { (byte) (this.position % PlaygroundConstants.COLUMN_COUNT + 'a'),
          (byte) (this.position / PlaygroundConstants.COLUMN_COUNT + '1') };
    else
      columnAsciiValues = new byte[] { (byte) (this.position % PlaygroundConstants.COLUMN_COUNT + 'A'),
          (byte) (this.position / PlaygroundConstants.COLUMN_COUNT + '1') };

    return new String(columnAsciiValues);
  }

  public boolean isValidMoveString(String moveString) {

    return Field.isValidFieldString(moveString);
  }
}
