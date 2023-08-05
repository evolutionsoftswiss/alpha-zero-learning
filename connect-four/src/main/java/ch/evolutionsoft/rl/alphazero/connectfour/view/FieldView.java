package ch.evolutionsoft.rl.alphazero.connectfour.view;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.geom.Ellipse2D;

import javax.swing.JPanel;

import ch.evolutionsoft.rl.alphazero.connectfour.model.ConnectFourGame;
import ch.evolutionsoft.rl.alphazero.connectfour.model.HumanPlayer;
import ch.evolutionsoft.rl.alphazero.connectfour.model.PlaygroundConstants;

public class FieldView extends JPanel implements MouseListener {

  private final int column;

  private int color = PlaygroundConstants.EMPTY;

  private transient ConnectFourGame game;

  private static final Color DARK_BLUE = new Color(0, 0, 200);
  private static final Color LIGHT_GREY = new Color(230, 230, 230);
  private static final Color MY_RED = new Color(220, 0, 0);
  private static final Color MY_YELLOW = new Color(225, 225, 0);

  public FieldView(ConnectFourGame game, int column) {

    super();
    this.game = game;
    this.column = column;
    this.setBackground(DARK_BLUE);
    this.addMouseListener(this);
  }

  public int getColor() {

    return this.color;
  }

  public void setColor(int color) {

    this.color = color;
  }

  public void reset() {

    this.color = PlaygroundConstants.EMPTY;
    this.repaint();
  }

  @Override
  protected void paintComponent(Graphics graphics) {

    super.paintComponent(graphics);
    Graphics2D graphics2D = (Graphics2D) graphics;

    int sideSize = this.getWidth();

    int gap = sideSize / 14;

    int circleSize = sideSize - 2 * gap;

    Ellipse2D.Double circle = new Ellipse2D.Double(gap, gap, circleSize, circleSize);
    if (color == PlaygroundConstants.EMPTY) {
      graphics2D.setColor(LIGHT_GREY);
    } else if (color == PlaygroundConstants.YELLOW) {
      graphics2D.setColor(MY_YELLOW);
    } else if (color == PlaygroundConstants.RED) {
      graphics2D.setColor(MY_RED);
    }
    graphics2D.fill(circle);
  }

  public void mouseClicked(MouseEvent mouseEvent) {

    if (game.getCurrentPlayer() instanceof HumanPlayer
        && game.notOver()) {
      ((HumanPlayer) game.getCurrentPlayer()).move(game, this.column);
    }
  }

  public void mousePressed(MouseEvent mouseEvent) {
    // No action
  }

  public void mouseReleased(MouseEvent mouseEvent) {
    // No action
  }

  public void mouseEntered(MouseEvent mouseEvent) {
    // No action
  }

  public void mouseExited(MouseEvent mouseEvent) {
    // No action
  }
}
