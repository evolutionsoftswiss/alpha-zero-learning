package ch.evolutionsoft.rl.alphazero.connectfour.controller;

import java.awt.event.ActionEvent;

import ch.evolutionsoft.rl.alphazero.connectfour.view.MainView;

public class PlayAction extends GameAction {

  public PlayAction(MainView mainView) {
    super("Play", mainView);
  }

  /*
   * (non-Javadoc)
   * 
   * @see
   * java.awt.event.ActionListener#actionPerformed(java.awt.event.ActionEvent)
   */
  public void actionPerformed(ActionEvent e) {
    this.getModel().play();
  }
}
