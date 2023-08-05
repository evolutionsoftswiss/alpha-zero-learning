package ch.evolutionsoft.rl.alphazero.connectfour.controller;

import java.awt.event.ActionEvent;

import ch.evolutionsoft.rl.alphazero.connectfour.view.MainView;

public class MonteCarloSearchOptionAction extends GameAction {

  public MonteCarloSearchOptionAction(MainView mainView) {
    super("MonteCarloSearchOptionAction", mainView);
  }

  @Override
  public void actionPerformed(ActionEvent arg0) {

    this.getModel().setUseMonteCarloSearch(this.mainView.monteCarloSearchOptionSelected());
  }

}
