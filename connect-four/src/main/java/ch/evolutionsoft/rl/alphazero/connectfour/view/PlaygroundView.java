package ch.evolutionsoft.rl.alphazero.connectfour.view;

import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.GridLayout;
import java.awt.event.ComponentAdapter;
import java.awt.event.ComponentEvent;
import java.util.Observable;
import java.util.Observer;

import javax.swing.JPanel;
import javax.swing.Timer;

import ch.evolutionsoft.rl.alphazero.connectfour.model.ConnectFourGame;
import ch.evolutionsoft.rl.alphazero.connectfour.model.Line;
import ch.evolutionsoft.rl.alphazero.connectfour.model.Move;
import ch.evolutionsoft.rl.alphazero.connectfour.model.PlaygroundConstants;

public class PlaygroundView extends JPanel implements Observer {

  private transient ConnectFourGame game;

  private FieldView[][] fieldViewElements = new FieldView[6][7];

  private Timer blinkWinningRowTimer;

  public static final int HALF_SECOND = 500;

  /**
   * 
   */
  public PlaygroundView(ConnectFourGame game) {

    super();
    this.game = game;

    this.game.addObserver(this);

    this.init();

    this.addComponentListener(new ComponentAdapter() {

      @Override
      public void componentResized(ComponentEvent arg0) {

        int w = getWidth() * PlaygroundConstants.ROW_COUNT;
        int h = getHeight() * PlaygroundConstants.COLUMN_COUNT;
        if (h <= w) {

          adaptSize(getHeight() / PlaygroundConstants.ROW_COUNT);
        } else {

          adaptSize(getWidth() / PlaygroundConstants.COLUMN_COUNT);
        }
        revalidate();
        repaint();
      }
    });
  }

  public void adaptSize(int sideSize) {

    for (int row = 5; row >= 0; row--) {
      for (int column = 0; column <= 6; column++) {
        fieldViewElements[row][column].setPreferredSize(new Dimension(sideSize, sideSize));
        fieldViewElements[row][column].revalidate();
      }
    }
  }

  public void blinkFourInARow() {

    Line winningRow = this.game.getWinningRow();

    final int rowDirection = winningRow.getWinningRowRowDirection();
    final int columnDirection = winningRow.getWinningRowColumnDirection();
    final int beginningRow = winningRow.getBeginning().getRow();
    final int beginningColumn = winningRow.getBeginning().getColumn();
    final int length = winningRow.getWinningRowLength();
    final int color = this.game.getWinner().getColor();

    this.blinkWinningRowTimer = new Timer(HALF_SECOND, actionListener -> 
        inverseWinningRow(beginningRow, beginningColumn, rowDirection, columnDirection, length, color)
    );

    this.blinkWinningRowTimer.start();
  }

  protected void inverseWinningRow(int beginningRow, int beginningColumn,
      int rowDirection, int columnDirection,
      int length, int color) {

    int inverseColor = (this.fieldViewElements[beginningRow][beginningColumn].getColor() == PlaygroundConstants.EMPTY)
        ? color
        : PlaygroundConstants.EMPTY;
    for (int count = 0; count < length; count++) {
      this.fieldViewElements[beginningRow + count * rowDirection][beginningColumn + count * columnDirection]
          .setColor(inverseColor);
    }
    this.repaint();
  }

  protected void resetWinningRow(int beginningRow, int beginningColumn, int rowDirection, int columnDirection,
      int length, int color) {

    int originalColor = color;
    for (int count = 0; count < length; count++) {
      this.fieldViewElements[beginningRow + count * rowDirection][beginningColumn + count * columnDirection]
          .setColor(originalColor);
    }
    this.repaint();
  }

  /**
   * 
   */
  private void init() {
    this.setLayout(new FlowLayout());

    JPanel centerPanel = new JPanel(new GridLayout(6, 7));
    for (int row = 5; row >= 0; row--)
      for (int column = 0; column <= 6; column++) {
        fieldViewElements[row][column] = new FieldView(game, column);
        centerPanel.add(fieldViewElements[row][column]);
      }

    this.add(centerPanel);
  }

  private void reset() {
 
    stopBlinkingWinningRow();
    for (int row = 0; row <= 5; row++) {
      for (int column = 0; column <= 6; column++) {
        fieldViewElements[row][column].reset();
      }
    }
  }

  private void stopBlinkingWinningRow() {

    if (this.blinkWinningRowTimer != null) {
      this.blinkWinningRowTimer.stop();
    }
  }

  /*
   * (non-Javadoc)
   * 
   * @see java.util.Observer#update(java.util.Observable, java.lang.Object)
   */
  public void update(Observable observable, Object object) {

    if ("New move".equals(object) || "Move redone".equals(object)) {

      Move move = this.game.getLastMove();
      fieldViewElements[move.getRow()][move.getColumn()].setColor(move.getColor());
      fieldViewElements[move.getRow()][move.getColumn()].repaint();

    } else if ("New game".equals(object)) {

      this.reset();

    } else if ("Move took back".equals(object)) {
 
      Line winningRow = this.game.getWinningRow();
      if (winningRow != null) {
        stopBlinkingWinningRow();
        this.resetWinningRow(
          winningRow.getBeginning().getRow(),
          winningRow.getBeginning().getColumn(),
          winningRow.getWinningRowRowDirection(),
          winningRow.getWinningRowColumnDirection(), 
          winningRow.getWinningRowLength(),
          winningRow.getColor());
        this.game.resetWinner();
      }
      Move move = this.game.getLastTookBackMove();
      fieldViewElements[move.getRow()][move.getColumn()].setColor(PlaygroundConstants.EMPTY);
      fieldViewElements[move.getRow()][move.getColumn()].repaint();
    }
  }
}