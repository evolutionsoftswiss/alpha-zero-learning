package ch.evolutionsoft.rl.alphazero.connectfour.controller;

import java.awt.event.ActionEvent;

import ch.evolutionsoft.rl.alphazero.connectfour.view.MainView;

public class ReDoMoveAction extends GameAction {

  public ReDoMoveAction(MainView mainView) {
    super("ReDoMoveAction", mainView);
  }

  /*
   * (non-Javadoc)
   * 
   * @see
   * java.awt.event.ActionListener#actionPerformed(java.awt.event.ActionEvent)
   */
  public void actionPerformed(ActionEvent e) {
    this.getModel().reDoMove();
  }

}
