package ch.evolutionsoft.rl.alphazero.connectfour.model;

public abstract class AbstractPlayer {

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
