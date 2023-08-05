package ch.evolutionsoft.rl.alphazero.connectfour.controller;

import javax.swing.AbstractAction;

import ch.evolutionsoft.rl.alphazero.connectfour.model.GameDriver;

import ch.evolutionsoft.rl.alphazero.connectfour.view.MainView;

public abstract class GameAction extends AbstractAction {

  protected MainView mainView;

  protected GameAction(String name, MainView mainView) {
    super(name);
    this.mainView = mainView;
  }

  protected GameDriver getModel() {
    return this.mainView.getModel();
  }
}
