package ch.evolutionsoft.rl4j.tictactoe;

import static ch.evolutionsoft.net.game.NeuralNetConstants.*;
import static ch.evolutionsoft.net.game.tictactoe.TicTacToeConstants.*;
import static ch.evolutionsoft.net.game.tictactoe.TicTacToeGameHelper.*;

import java.util.HashSet;
import java.util.Set;

import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.learning.NeuralNetFetchable;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.deeplearning4j.rl4j.space.ArrayObservationSpace;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.ObservationSpace;
import org.json.JSONObject;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ch.evolutionsoft.rl4j.tictactoe.ReinforcementLearningMain.TicTacToeAction;
import ch.evolutionsoft.rl4j.tictactoe.ReinforcementLearningMain.TicTacToeState;

public class TicTacToeGame implements MDP<TicTacToeState, Integer, DiscreteSpace> {

  /**
   * Set draw reward nearer MAX_WIN_REWARD as its desired in perfect play
   */
  static final double DRAW_REWARD = 0.9;

  static final double MAX_WIN_REWARD = 1;

  static final double MIN_WIN_REWARD = 0;
  
  double trainingPlayer = MAX_PLAYER;
  
  boolean enableAssertions = false;

  private static final Logger logger = LoggerFactory.getLogger(TicTacToeGame.class);

  TicTacToeState currentState = new TicTacToeState(ReinforcementLearningMain.EMPTY_CONVOLUTIONAL_PLAYGROUND, 0);
  
  NeuralNetFetchable<IDQN<ConvolutionalNeuralNetDQN>> fetchable;
  
  TicTacToeGame() {
  }

  public ObservationSpace<TicTacToeState> getObservationSpace() {

    return new ArrayObservationSpace<>(
      new int[] {1, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE }
    );
  }

  public INDArray getCurrentPlayground() {
    
    return currentState.getPlayground();
  }
  
  @Override
  public TicTacToeAction getActionSpace() {

    Set<Integer> availableMoves = getEmptyFields(currentState.getPlayground());

    TicTacToeAction currentActions = new TicTacToeAction(availableMoves);

    invariant();
    
    printTest(availableMoves);
    
    return currentActions;
  }

  public TicTacToeAction getActionSpace(INDArray inputState) {

    Set<Integer> availableMoves = getEmptyFields(inputState);

    TicTacToeAction currentActions = new TicTacToeAction(availableMoves);

    invariant();
    
    printTest(availableMoves);
    
    return currentActions;
  }

  public TicTacToeState reset() {
    
    currentState = new TicTacToeState(ReinforcementLearningMain.EMPTY_CONVOLUTIONAL_PLAYGROUND, 0);
    return currentState;
  }

  @Override
  public MDP<TicTacToeState, Integer, DiscreteSpace> newInstance() {
    return new TicTacToeGame();
  }

  public void close() {
    // Intentionally blank atm
  }

  @Override
  public StepReply<TicTacToeState> step(Integer action) {

    invariant();
    
    if (Integer.valueOf(-1).equals(action)) {

      throw new IllegalStateException();
    }

    currentState = currentState.makeMove(action);
    
    double reward = calculateReward(currentState);

    invariant();

    printDebugTest();
    
    return new StepReply<>(currentState, reward, isDone(), new JSONObject("{}"));
  }

  protected double calculateReward(TicTacToeState newTicTacToeState) {

    INDArray newPlayground = newTicTacToeState.getPlayground();

    if (logger.isDebugEnabled()) {
      logger.debug(String.valueOf(newPlayground));
    }
    
    if (MAX_PLAYER == trainingPlayer) {
      if (hasWon(MAX_PLAYER_CHANNEL)) {
  
        return MAX_WIN_REWARD;
      }
  
      if (hasWon(MIN_PLAYER_CHANNEL)) {
        
        return MIN_WIN_REWARD;
      }
    
    } else if (hasWon(MAX_PLAYER_CHANNEL)) {
        
      return MIN_WIN_REWARD;
    
    } else if (hasWon(MIN_PLAYER_CHANNEL)) {
  
      return MAX_WIN_REWARD;
    }    

    return DRAW_REWARD;
  }

  public double switchTrainingPlayer() {
    
    this.setTrainingPlayer(this.trainingPlayer == MAX_PLAYER ? MIN_PLAYER : MAX_PLAYER);
    
    return this.getTrainingPlayer();
  }
  
  public boolean isDone() {

    return noEmptyFieldsLeft(currentState.getPlayground()) || hasWon(MAX_PLAYER_CHANNEL) || hasWon(MIN_PLAYER_CHANNEL);
  }

  boolean noEmptyFieldsLeft(INDArray inputState) {

    //Performance possible
    return getEmptyFields(inputState).isEmpty();
  }
  
  Set<Integer> getEmptyFields(INDArray playground) {
    
    Set<Integer> emptyFieldsIndices = new HashSet<>(SMALL_CAPACITY);
    
    for (int row = 0; row < IMAGE_SIZE; row++) {
      for (int column = 0; column < IMAGE_SIZE; column++) {
  
        if (equalsEpsilon(OCCUPIED_IMAGE_POINT,
            playground.getDouble(EMPTY_FIELDS_CHANNEL, row, column),
            DOUBLE_COMPARISON_EPSILON) ) {
  
          emptyFieldsIndices.add(IMAGE_SIZE * row + column);
        }
      }
    }
    
    return emptyFieldsIndices;
  }
  
  double getCurrentPlayer(Set<Integer> emptyFields) {
    
    boolean oddOccupiedFields = 1 == countOccupiedImagePoints(emptyFields.size()) % 2;
    
    if (oddOccupiedFields) {
      
      return MIN_PLAYER;
    }
    
    return MAX_PLAYER;
  }

  boolean hasWon(int player) {

    return horizontalWin(player) || verticalWin(player) || diagonalWin(player);
  }

  boolean horizontalWin(int player) {
    
    INDArray playerFieldsImage = currentState.getPlayground();

    return (playerFieldsImage.getDouble(player, 0, 0) == OCCUPIED_IMAGE_POINT &&
        playerFieldsImage.getDouble(player, 0, 1) == OCCUPIED_IMAGE_POINT &&
            playerFieldsImage.getDouble(player, 0, 2) == OCCUPIED_IMAGE_POINT) ||
           (playerFieldsImage.getDouble(player, 1, 0) == OCCUPIED_IMAGE_POINT &&
               playerFieldsImage.getDouble(player, 1, 1) == OCCUPIED_IMAGE_POINT &&
                   playerFieldsImage.getDouble(player, 1, 2) == OCCUPIED_IMAGE_POINT) ||
           (playerFieldsImage.getDouble(player, 2, 0) == OCCUPIED_IMAGE_POINT &&
               playerFieldsImage.getDouble(player, 2, 1) == OCCUPIED_IMAGE_POINT &&
                   playerFieldsImage.getDouble(player, 2, 2) == OCCUPIED_IMAGE_POINT);
  }

  boolean diagonalWin(int player) {
    
    INDArray playerFieldsImage = currentState.getPlayground();

    return (playerFieldsImage.getDouble(player, 0, 0) == OCCUPIED_IMAGE_POINT &&
        playerFieldsImage.getDouble(player, 1, 1) == OCCUPIED_IMAGE_POINT &&
            playerFieldsImage.getDouble(player, 2, 2) == OCCUPIED_IMAGE_POINT) ||
           (playerFieldsImage.getDouble(player, 0, 2) == OCCUPIED_IMAGE_POINT &&
               playerFieldsImage.getDouble(player, 1, 1) == OCCUPIED_IMAGE_POINT &&
                   playerFieldsImage.getDouble(player, 2, 0) == OCCUPIED_IMAGE_POINT);
  }

  boolean verticalWin(int player) {
    
    INDArray playerFieldsImage = currentState.getPlayground();

    return (playerFieldsImage.getDouble(player, 0, 0) == OCCUPIED_IMAGE_POINT &&
        playerFieldsImage.getDouble(player, 1, 0) == OCCUPIED_IMAGE_POINT &&
            playerFieldsImage.getDouble(player, 2, 0) == OCCUPIED_IMAGE_POINT) ||
           (playerFieldsImage.getDouble(player, 0, 1) == OCCUPIED_IMAGE_POINT &&
               playerFieldsImage.getDouble(player, 1, 1) == OCCUPIED_IMAGE_POINT &&
                   playerFieldsImage.getDouble(player, 2, 1) == OCCUPIED_IMAGE_POINT) ||
           (playerFieldsImage.getDouble(player, 0, 2) == OCCUPIED_IMAGE_POINT &&
               playerFieldsImage.getDouble(player, 1, 2) == OCCUPIED_IMAGE_POINT &&
                   playerFieldsImage.getDouble(player, 2, 2) == OCCUPIED_IMAGE_POINT);
  }
  
  /**
   * returns the number of empty points for empty point channel.
   * 
   * @param channelImage
   * @return number of occupied 1 image points
   */
  int countOccupiedImagePoints(int emptyFieldsCount) {
    
    return COLUMN_COUNT - emptyFieldsCount;
  }

  public void printTest() {

    INDArray output = fetchable.getNeuralNet().output(
        currentState.getPlayground());
    if (logger.isInfoEnabled()) {
      logger.info(String.valueOf(currentState.getPlayground()));
      logger.info(String.valueOf(output));
    }
  }

  public void printDebugTest() {

    if (logger.isDebugEnabled() && null != fetchable) {

      INDArray output = fetchable.getNeuralNet().output(
          currentState.getPlayground());
      logger.info(String.valueOf(currentState.getPlayground()));
      logger.info(String.valueOf(output));
    }
  }

  public void printTest(Set<Integer> availableMoves) {

    if (logger.isDebugEnabled()) {

      logger.info(String.valueOf(availableMoves));
    }
  }

  public void invariant() {

    if (enableAssertions) {
      INDArray playground = getCurrentPlayground();
      int maxStones = countMaxStones(playground);
      int minStones = countMinStones(playground);
      int emptyFieldsCount = getEmptyFields(currentState.getPlayground()).size();
      int totalStones = countOccupiedImagePoints(emptyFieldsCount);
      
      assert maxStones == minStones || maxStones == minStones + 1;
      assert totalStones == maxStones + minStones;
      assert emptyFieldsCount == IMAGE_POINTS - totalStones;
    }
  }
  
  public NeuralNetFetchable<IDQN<ConvolutionalNeuralNetDQN>> getFetchable() {
    return fetchable;
  }

  public void setFetchable(NeuralNetFetchable<IDQN<ConvolutionalNeuralNetDQN>> fetchable) {
    this.fetchable = fetchable;
  }
  
  public TicTacToeState getLastObs() {
    return currentState;
  }

  public void setCurrentState(TicTacToeState currentState) {
    this.currentState = currentState;
  }

  public double getTrainingPlayer() {
    return trainingPlayer;
  }

  public void setTrainingPlayer(double trainingPlayer) {
    this.trainingPlayer = trainingPlayer;
  }

  public int getSteps() {
    
    return currentState.depth;
  }
}
