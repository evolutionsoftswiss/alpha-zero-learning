package ch.evolutionsoft.rl.alphazero.connectfour.view;

import java.awt.BorderLayout;
import java.awt.Font;
import java.awt.GridLayout;

import javax.swing.JButton;
import javax.swing.JCheckBoxMenuItem;
import javax.swing.JFrame;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JPanel;
import javax.swing.KeyStroke;

import ch.evolutionsoft.rl.alphazero.connectfour.model.GameDriver;
import ch.evolutionsoft.rl.alphazero.connectfour.controller.*;

public class MainView extends JFrame {

  private transient GameDriver gameDriver;

  private JCheckBoxMenuItem twoPlayerCheckBoxMenuItem;
  
  private JCheckBoxMenuItem monteCarloSearchOptionMenuItem;

  protected Font buttonFont = new Font("Arial", Font.BOLD, 15);
 
  public static void main(String[] args) {
    
    MainView mainView = new MainView();
    
    mainView.init();

    mainView.setVisible(true);
  }

  public void init() {

    this.initMenuBar();
    this.getContentPane().setLayout(new BorderLayout());
    this.getContentPane().add(this.makeButtonPanel(), BorderLayout.NORTH);
    this.gameDriver = new GameDriver();
    this.getContentPane().add(new GameView(gameDriver.getGame()), BorderLayout.CENTER);
    this.setSize(430, 480);
    this.setVisible(true);
  }

  public GameDriver getModel() {

    return this.gameDriver;
  }

  public boolean monteCarloSearchOptionSelected() {
    
    return this.monteCarloSearchOptionMenuItem.isSelected();
  }
  
  public boolean twoPlayersSelected() {

    return this.twoPlayerCheckBoxMenuItem.isSelected();
  }

  protected void initMenuBar() {

    JMenuBar menuBar = new JMenuBar();
    menuBar.add(this.makeCommandMenu());
    menuBar.add(this.makeOptionMenu());
    this.setJMenuBar(menuBar);
  }

  protected JPanel makeButtonPanel() {

    JPanel buttonPanel = new JPanel(new GridLayout(1, 5));
    JButton newGameButton = new JButton(new NewGameAction(this));
    newGameButton.setText("New");
    newGameButton.setFont(this.buttonFont);

    JButton playButton = new JButton(new PlayAction(this));
    playButton.setText("Go");
    playButton.setFont(this.buttonFont);

    JButton takeBackButton = new JButton(new TakeBackAction(this));
    takeBackButton.setText("\u2190");
    takeBackButton.setFont(this.buttonFont);

    JButton reDoButton = new JButton(new ReDoMoveAction(this));
    reDoButton.setText("\u2192");
    reDoButton.setFont(this.buttonFont);

    buttonPanel.add(newGameButton);
    buttonPanel.add(playButton);
    buttonPanel.add(new JPanel());// add empty Panel to have empty GridCell
    buttonPanel.add(takeBackButton);
    buttonPanel.add(reDoButton);

    return buttonPanel;
  }

  protected JMenu makeCommandMenu() {

    JMenu fileMenu = new JMenu("Command");
    JMenuItem newGameMenuItem = new JMenuItem(new NewGameAction(this));
    newGameMenuItem.setText("New");
    newGameMenuItem.setAccelerator(KeyStroke.getKeyStroke("F2"));

    JMenuItem playMenuItem = new JMenuItem(new PlayAction(this));
    playMenuItem.setText("Do move (Go)");
    playMenuItem.setAccelerator(KeyStroke.getKeyStroke("F3"));

    JMenuItem takeBackMenuItem = new JMenuItem(new TakeBackAction(this));
    takeBackMenuItem.setText("Undo move (\u2190)");
    takeBackMenuItem.setAccelerator(KeyStroke.getKeyStroke("F5"));

    JMenuItem reDoMenuItem = new JMenuItem(new ReDoMoveAction(this));
    reDoMenuItem.setText("Redo move (\u2192)");
    reDoMenuItem.setAccelerator(KeyStroke.getKeyStroke("F6"));

    fileMenu.add(newGameMenuItem);
    fileMenu.add(playMenuItem);
    fileMenu.add(takeBackMenuItem);
    fileMenu.add(reDoMenuItem);
    return fileMenu;
  }

  protected JMenu makeOptionMenu() {

    JMenu optionsMenu = new JMenu("Options");
    this.initMonteCarloSearchOptionMenuItem();
    this.initTwoPlayerMenuItem();
    optionsMenu.add(this.monteCarloSearchOptionMenuItem);
    optionsMenu.add(this.twoPlayerCheckBoxMenuItem);
    return optionsMenu;
  }

  protected void initMonteCarloSearchOptionMenuItem() {

    this.monteCarloSearchOptionMenuItem = new JCheckBoxMenuItem(new MonteCarloSearchOptionAction(this));
    this.monteCarloSearchOptionMenuItem.setSelected(true);
    this.monteCarloSearchOptionMenuItem.setText("Use additional Monte Carlo Search instead of Neural net output only");
  }

  protected void initTwoPlayerMenuItem() {

    this.twoPlayerCheckBoxMenuItem = new JCheckBoxMenuItem(new ChangeNumberOfPlayersAction(this));
    this.twoPlayerCheckBoxMenuItem.setText("Two Players");
  }
}
