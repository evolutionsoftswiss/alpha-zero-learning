package ch.evolutionsoft.rl4j;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ch.evolutionsoft.net.game.NeuralNetConstants;
import ch.evolutionsoft.net.game.tictactoe.TicTacToeConstants;
import ch.evolutionsoft.rl4j.tictactoe.TicTacToe;
import ch.evolutionsoft.rl4j.tictactoe.TicTacToeStateAction;

public class MonteCarloTreeSearch {

  private static final Logger log = LoggerFactory.getLogger(MonteCarloTreeSearch.class);

  Map<TicTacToeStateAction, Float> qValueByBoardAction = new HashMap<>();

  Map<TicTacToeStateAction, Integer> visitedTimesByBoardAction = new HashMap<>();

  Map<INDArray, Integer> visitedTimesByBoard = new HashMap<>();

  Map<INDArray, INDArray> initialQValuesByBoard = new HashMap<>();

  Map<INDArray, Float> gameEndedByBoard = new HashMap<>();

  Map<INDArray, INDArray> validMovesByBoard = new HashMap<>();
  
  float cUct = 1f;
  
  int numberOfSimulations = 30;

  ComputationGraph computationGraph;

  public MonteCarloTreeSearch(ComputationGraph computationGraph) {

    this.computationGraph = computationGraph;
  }

  public INDArray getActionValues(INDArray board, float temperature) {

    for (int simulationNumber = 1; simulationNumber < numberOfSimulations; simulationNumber++) {
      
      this.monteCarloSearch(board.dup());
    }

    int[] visitedCounts = new int[TicTacToeConstants.COLUMN_COUNT];

    float totalVisitedCount = 0;
    for (int index = 0; index < TicTacToeConstants.COLUMN_COUNT; index++) {
      
      TicTacToeStateAction currentStateAction = new TicTacToeStateAction(board, index);
      if (this.visitedTimesByBoardAction.containsKey(currentStateAction)) {
        
        visitedCounts[index] = this.visitedTimesByBoardAction.get(currentStateAction);
      
      } else {
        
        visitedCounts[index] = 0;
      }
 
      totalVisitedCount += visitedCounts[index];
    }
    
    INDArray moveProbabilities = Nd4j.zeros(TicTacToeConstants.COLUMN_COUNT);

    if (0 == temperature) {
      
      INDArray visitedCountsArray = Nd4j.createFromArray(visitedCounts);
      
      INDArray visitedCountsMax = visitedCountsArray.argMax(0);// Lower index is taken on equal max values
      
      moveProbabilities.putScalar(visitedCountsMax.getInt(0), 1f);
      
      return moveProbabilities;
    }
    
    for (int index = 0; index < TicTacToeConstants.COLUMN_COUNT; index++) {
      
      moveProbabilities.putScalar(index, Math.pow(visitedCounts[index], 1.0 / temperature) / totalVisitedCount);
    }
    
    return moveProbabilities;
  }

  float monteCarloSearch(INDArray board) {

    Set<Integer> emptyFields = TicTacToe.getEmptyFields(board);

    if (!this.gameEndedByBoard.containsKey(board)) {

      int otherPlayer = (int) TicTacToe.getOtherPlayer(emptyFields);
      if (TicTacToe.hasWon(board, otherPlayer)) {

        float boardValue = otherPlayer == TicTacToeConstants.MIN_PLAYER_CHANNEL ? -1f : 1f;
        this.gameEndedByBoard.put(board, boardValue);

        return -boardValue;

      } 
      
      if (emptyFields.isEmpty()) {

        float drawValue = 0.001f;
        this.gameEndedByBoard.put(board, drawValue);

        return drawValue;
      }
    }
    
    if (this.gameEndedByBoard.containsKey(board)) {
      
      return -this.gameEndedByBoard.get(board);
    }
    
    INDArray availableMoves = TicTacToe.getValidMoves(board);

    if (!initialQValuesByBoard.containsKey(board)) {

      INDArray oneBatchBoard = Nd4j.zeros(1, 3, 3, 3);
      oneBatchBoard.putRow(0, board);

      INDArray[] nnOutput = this.computationGraph.output(oneBatchBoard);
      INDArray validProbabilities = nnOutput[0].mul(availableMoves);
      this.initialQValuesByBoard.put(board, validProbabilities);
      float currentBoardValue = nnOutput[1].getFloat(0);

      float qValuesTotal = this.initialQValuesByBoard.get(board).sumNumber().floatValue();

      if (qValuesTotal > 0) {
        INDArray unnormalizedQValues = this.initialQValuesByBoard.remove(board);
        this.initialQValuesByBoard.put(board, unnormalizedQValues.div(qValuesTotal));

      } else {

        INDArray invalidProbableMoves = this.initialQValuesByBoard.remove(board);
        INDArray validMovesNormalized = invalidProbableMoves.add(availableMoves).div(availableMoves.sumNumber());
        this.initialQValuesByBoard.put(board, validMovesNormalized);
        log.warn("All valid moves had zero probability, set valid moves equally probable {}", validMovesNormalized);
      }

      this.validMovesByBoard.put(board, availableMoves);
      this.visitedTimesByBoard.put(board, 0);

      return -currentBoardValue;
    }

    availableMoves = this.validMovesByBoard.get(board);
    float bestMoveValue = Float.NEGATIVE_INFINITY;
    Integer bestMoveIndex = -1;
    
    for (int index = 0; index < TicTacToeConstants.COLUMN_COUNT; index++) {
      
      if (availableMoves.getFloat(index) > NeuralNetConstants.DOUBLE_COMPARISON_EPSILON) {
        
        TicTacToeStateAction availableMove = new TicTacToeStateAction(board, index);
        float currentMoveValue;

        if (this.qValueByBoardAction.containsKey(availableMove)) {
          
          currentMoveValue = (float) (this.qValueByBoardAction.get(availableMove) + 
              this.cUct * this.initialQValuesByBoard.get(board).getFloat(index) * Math.sqrt(this.visitedTimesByBoard.get(board)) /
              (1 + this.visitedTimesByBoardAction.get(availableMove)));
          
        } else {

          currentMoveValue = (float) (this.cUct * this.initialQValuesByBoard.get(board).getFloat(index) *
              Math.sqrt(this.visitedTimesByBoard.get(board) + NeuralNetConstants.DOUBLE_COMPARISON_EPSILON));
        }
        
        if (currentMoveValue > bestMoveValue) {
          
          bestMoveValue = currentMoveValue;
          bestMoveIndex = index;
        }
      }
    }
    
    Integer moveIndex = bestMoveIndex;
    int currentPlayer = TicTacToe.getCurrentPlayer(emptyFields);
    INDArray nextBoard = TicTacToe.makeMove(board, moveIndex, currentPlayer);
    
    float value = this.monteCarloSearch(nextBoard);
    
    TicTacToeStateAction lastAction = new TicTacToeStateAction(board, moveIndex);

    if (this.qValueByBoardAction.containsKey(lastAction)) {
      
      int visitedTimesBoardBefore = this.visitedTimesByBoardAction.get(lastAction);
      this.qValueByBoardAction.put(lastAction,
          (visitedTimesBoardBefore * this.qValueByBoardAction.get(lastAction) + value) /
          (visitedTimesBoardBefore + 1)
          );
      this.visitedTimesByBoardAction.put(lastAction, visitedTimesBoardBefore + 1);
    
    } else {
      
      this.qValueByBoardAction.put(lastAction, value);
      this.visitedTimesByBoardAction.put(lastAction, 1);
    }
    
    this.visitedTimesByBoard.put(board, visitedTimesByBoard.get(board) + 1);
    
    return -value;
  }
}
