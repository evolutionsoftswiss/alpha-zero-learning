package ch.evolutionsoft.rl.alphazero.connectfour.model;

import java.util.Observable;

public abstract class AbstractPlayer extends Observable {

  protected int color;

  protected AbstractPlayer(int color) {
    this.color = color;
  }

  public int getColor() {
    return this.color;
  }

  public void setColor(int color) {
    this.color = color;
  }
}
