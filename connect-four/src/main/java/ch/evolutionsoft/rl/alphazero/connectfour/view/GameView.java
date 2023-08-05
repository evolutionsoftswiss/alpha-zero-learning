package ch.evolutionsoft.rl.alphazero.connectfour.view;

import java.awt.BorderLayout;
import java.awt.Font;
import java.util.Observable;
import java.util.Observer;

import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.SwingConstants;

import ch.evolutionsoft.rl.alphazero.connectfour.model.ConnectFourGame;

public class GameView extends JPanel implements Observer {

  protected transient ConnectFourGame game;

  protected PlaygroundView playgroundView;

  protected JLabel informationLabel = new JLabel();

  protected Font informationFont = new Font("Arial", Font.BOLD, 17);

  public GameView(ConnectFourGame game) {

    super();
    this.game = game;
    this.game.addObserver(this);

    this.addObserverToComputerPlayer();

    this.playgroundView = new PlaygroundView(game);

    this.initComponents();
  }

  private void addObserverToComputerPlayer() {

    if (this.game.getComputerPlayer() != null) {

      this.game.getComputerPlayer().addObserver(this);
    }
  }

  private void initComponents() {

    this.setLayout(new BorderLayout());

    this.add(this.playgroundView, BorderLayout.CENTER);

    informationLabel = new JLabel("Yellow to move", SwingConstants.CENTER);
    informationLabel.setFont(this.informationFont);
    this.add(informationLabel, BorderLayout.SOUTH);
  }

  /*
   * (non-Javadoc)
   * 
   * @see java.util.Observer#update(java.util.Observable, java.lang.Object)
   */
  public void update(Observable arg0, Object arg1) {

    if (((String) arg1).equals("New game")
        || ((String) arg1).equals("Other player set")) {

      this.addObserverToComputerPlayer();
    }

    else if (((String) arg1).equals("Four in a row")) {

      this.playgroundView.blinkFourInARow();
    }

    informationLabel.setText(this.game.getGameState());
    this.repaint();
  }
}
