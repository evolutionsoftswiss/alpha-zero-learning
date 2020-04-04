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

  static final double DRAW_REWARD = 0.5;

  static final double MAX_WIN_REWARD = 1;

  static final double MIN_WIN_REWARD = 0;
  
  static final double TRAINING_PLAYER = MAX_PLAYER;

  private static final Logger logger = LoggerFactory.getLogger(TicTacToeGame.class);

  TicTacToeState currentState = new TicTacToeState(EMPTY_PLAYGROUND, 0);

  double accumulatedReward;
  
  NeuralNetFetchable<IDQN> fetchable;
  
  TicTacToeGame() {
  }
  
  TicTacToeGame(int step, TicTacToeState nextO, double reward) {
    
    currentState = nextO;
    currentState.depth = step;
    accumulatedReward = reward;
  }

  public ObservationSpace<TicTacToeState> getObservationSpace() {

    ArrayObservationSpace<TicTacToeState> observationSpace =
        new ArrayObservationSpace<>(new int[] { COLUMN_COUNT });

    return observationSpace;
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
    
    return currentState = new TicTacToeState(EMPTY_PLAYGROUND, 0);
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
    
    return new StepReply<TicTacToeState>(currentState, reward, isDone(), new JSONObject("{}"));
  }

  protected double calculateReward(TicTacToeState newTicTacToeState) {

    INDArray newPlayground = newTicTacToeState.getPlayground();

    if (hasWon(MAX_PLAYER_CHANNEL)) {

        logger.debug(String.valueOf(newPlayground));
        return MAX_WIN_REWARD; // + 0.1 * this.currentState.depth;
      }

      if (hasWon(MIN_PLAYER_CHANNEL)) {

        logger.debug(String.valueOf(newPlayground));
        return MIN_WIN_REWARD; // - 0.1 * this.currentState.depth;

      }
      
      if (noEmptyFieldsLeft(newPlayground)) {

        logger.debug(String.valueOf(newPlayground));
        return DRAW_REWARD;
      }
      
      return DRAW_REWARD;
  }

  public boolean isDone() {

    return noEmptyFieldsLeft(currentState.getPlayground()) || hasWon(MAX_PLAYER_CHANNEL) || hasWon(MIN_PLAYER_CHANNEL);
  }

  boolean noEmptyFieldsLeft(INDArray inputState) {

    //Performance possible
    return getEmptyFields(inputState).isEmpty();
  }
  
  boolean allFieldsEmpty(INDArray inputState) {
  
    //Performance possible
    return IMAGE_POINTS == getEmptyFields(inputState).size();
  }
  
  Set<Integer> getEmptyFields(INDArray inputState) {
    
    Set<Integer> emptyFieldsIndices = new HashSet<>(SMALL_CAPACITY);
    for (int column = 0; column < COLUMN_COUNT; column++) {

      if (equalsEpsilon(EMPTY_FIELD_VALUE, inputState.getDouble(column), DOUBLE_COMPARISON_EPSILON) ) {

        emptyFieldsIndices.add(column);
      }
    }
    
    return emptyFieldsIndices;
  }
  
  int getCurrentPlayerChannel(INDArray inputState) {
    
    boolean oddEmptyFields = 1 == countOccupiedImagePoints(inputState) % 2;
    
    if (oddEmptyFields) {
      
      return MIN_PLAYER_CHANNEL;
    }
    
    return MAX_PLAYER_CHANNEL;
  }

  boolean hasWon(int playerChannel) {

    return horizontalWin(playerChannel) || verticalWin(playerChannel) || diagonalWin(playerChannel);
  }

  boolean horizontalWin(int playerChannel) {
    
    INDArray playerFieldsImage = currentState.getPlayground();

    return (playerFieldsImage.getDouble(FIELD_1) == playerChannel &&
        playerFieldsImage.getDouble(FIELD_2) == playerChannel &&
            playerFieldsImage.getDouble(FIELD_3) == playerChannel) ||
           (playerFieldsImage.getDouble(FIELD_4) == playerChannel &&
               playerFieldsImage.getDouble(FIELD_5) == playerChannel &&
                   playerFieldsImage.getDouble(FIELD_6) == playerChannel) ||
           (playerFieldsImage.getDouble(FIELD_7) == playerChannel &&
               playerFieldsImage.getDouble(FIELD_8) == playerChannel &&
                   playerFieldsImage.getDouble(FIELD_9) == playerChannel);
  }

  boolean diagonalWin(int playerChannel) {
    
    INDArray playerFieldsImage = currentState.getPlayground();

    return (playerFieldsImage.getDouble(FIELD_1) == playerChannel &&
        playerFieldsImage.getDouble(FIELD_5) == playerChannel &&
            playerFieldsImage.getDouble(FIELD_9) == playerChannel) ||
           (playerFieldsImage.getDouble(FIELD_3) == playerChannel &&
               playerFieldsImage.getDouble(FIELD_5) == playerChannel &&
                   playerFieldsImage.getDouble(FIELD_7) == playerChannel);
  }

  boolean verticalWin(int playerChannel) {
    
    INDArray playerFieldsImage = currentState.getPlayground();

    return (playerFieldsImage.getDouble(FIELD_1) == playerChannel &&
        playerFieldsImage.getDouble(FIELD_4) == playerChannel &&
            playerFieldsImage.getDouble(FIELD_7) == playerChannel) ||
           (playerFieldsImage.getDouble(FIELD_2) == playerChannel &&
               playerFieldsImage.getDouble(FIELD_5) == playerChannel &&
                   playerFieldsImage.getDouble(FIELD_8) == playerChannel) ||
           (playerFieldsImage.getDouble(FIELD_3) == playerChannel &&
               playerFieldsImage.getDouble(FIELD_6) == playerChannel &&
                   playerFieldsImage.getDouble(FIELD_9) == playerChannel);
  }
  
  /**
   * returns the number of empty points for empty point channel.
   * 
   * @param channelImage
   * @return number of occupied 1 image points
   */
  int countOccupiedImagePoints(INDArray channelImage) {
    
    return COLUMN_COUNT - getEmptyFields(channelImage).size();
  }

  public void printTest() {

    INDArray output = fetchable.getNeuralNet().output(
        currentState.getPlayground());
    logger.info(String.valueOf(currentState.getPlayground()));
    logger.info(String.valueOf(output));
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

    INDArray playground = getCurrentPlayground();
    int maxStones = countMaxStones(playground);
    int minStones = countMinStones(playground);
    int emptyFieldsCount = getEmptyFields(currentState.getPlayground()).size();
    int totalStones = COLUMN_COUNT - countOccupiedImagePoints(currentState.getPlayground());
    
    assert maxStones == minStones || maxStones == minStones + 1;
    assert totalStones == maxStones + minStones;
    assert emptyFieldsCount == IMAGE_POINTS - totalStones;
  }
  
  public NeuralNetFetchable<IDQN> getFetchable() {
    return fetchable;
  }

  public void setFetchable(NeuralNetFetchable<IDQN> fetchable) {
    this.fetchable = fetchable;
  }
  
  public TicTacToeState getLastObs() {
    return currentState;
  }

  public void setCurrentState(TicTacToeState currentState) {
    this.currentState = currentState;
  }

  public double getReward() {
    return accumulatedReward;
  }

  public void setReward(double accumulatedReward) {
    this.accumulatedReward = accumulatedReward;
  }

  public int getSteps() {
    
    return currentState.depth;
  }
}
