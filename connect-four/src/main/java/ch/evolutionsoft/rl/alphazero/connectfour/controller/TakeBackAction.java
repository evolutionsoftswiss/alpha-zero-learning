package ch.evolutionsoft.rl.alphazero.connectfour.controller;

import java.awt.event.ActionEvent;

import ch.evolutionsoft.rl.alphazero.connectfour.view.MainView;

public class TakeBackAction extends GameAction {

  public TakeBackAction(MainView mainView) {
    super("Take back", mainView);
  }

  /*
   * (non-Javadoc)
   * 
   * @see
   * java.awt.event.ActionListener#actionPerformed(java.awt.event.ActionEvent)
   */
  public void actionPerformed(ActionEvent arg0) {
    this.getModel().takeBackMove();
  }

}
