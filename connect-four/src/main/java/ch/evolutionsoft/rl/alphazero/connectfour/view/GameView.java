package ch.evolutionsoft.rl.alphazero.connectfour.view;

import static ch.evolutionsoft.rl.alphazero.connectfour.model.ModelViewConstants.*;

import java.awt.BorderLayout;
import java.awt.Font;
import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;

import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.SwingConstants;

import ch.evolutionsoft.rl.alphazero.connectfour.model.ConnectFourGame;

public class GameView extends JPanel implements PropertyChangeListener {

  protected transient ConnectFourGame game;

  protected PlaygroundView playgroundView;

  protected JLabel informationLabel = new JLabel();

  protected Font informationFont = new Font("Arial", Font.BOLD, 17);

  public GameView(ConnectFourGame game) {

    super();
    this.game = game;
    this.game.addPropertyChangeListener(this);

    addPropertyListenerToComputerPlayer();

    this.playgroundView = new PlaygroundView(game);

    this.initComponents();
  }

  @Override
  public void propertyChange(PropertyChangeEvent event) {

    if (NEW_GAME_PROPERTY.equals(event.getPropertyName())
        || OTHER_PLAYER_PROPERTY.equals(event.getPropertyName())) {

      this.addPropertyListenerToComputerPlayer();
    }

    informationLabel.setText(this.game.getGameState());
    this.repaint();
  }

  protected void addPropertyListenerToComputerPlayer() {

    if (this.game.getComputerPlayer() != null) {

      this.game.getComputerPlayer().addPropertyChangeListener(this);
    }
  }

  private void initComponents() {

    this.setLayout(new BorderLayout());

    this.add(this.playgroundView, BorderLayout.CENTER);

    informationLabel = new JLabel("Yellow to move", SwingConstants.CENTER);
    informationLabel.setFont(this.informationFont);
    this.add(informationLabel, BorderLayout.SOUTH);
  }
}
