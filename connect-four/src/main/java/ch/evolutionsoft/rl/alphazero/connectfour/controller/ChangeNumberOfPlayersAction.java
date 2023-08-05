package ch.evolutionsoft.rl.alphazero.connectfour.controller;

import java.awt.event.ActionEvent;

import ch.evolutionsoft.rl.alphazero.connectfour.model.GameDriver;

import ch.evolutionsoft.rl.alphazero.connectfour.view.MainView;

public class ChangeNumberOfPlayersAction extends GameAction {

  public ChangeNumberOfPlayersAction(MainView mainView) {
    super("ChangeNumberOfPlayersAction", mainView);
  }

  /*
   * (non-Javadoc)
   * 
   * @see
   * java.awt.event.ActionListener#actionPerformed(java.awt.event.ActionEvent)
   */
  public void actionPerformed(ActionEvent e) {
    if (this.mainView.twoPlayersSelected())
      this.getModel().setNumberOfPlayers(GameDriver.TWO_PLAYERS);
    else
      this.getModel().setNumberOfPlayers(GameDriver.ONE_PLAYER);
  }

}
