package ch.evolutionsoft.rl.alphazero;

import java.util.Arrays;
import java.util.Set;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ch.evolutionsoft.rl.AdversaryLearningConfiguration;
import ch.evolutionsoft.rl.AdversaryLearningConstants;
import ch.evolutionsoft.rl.Game;

public class MonteCarloTreeSearch {

  Logger logger = LoggerFactory.getLogger(MonteCarloTreeSearch.class);

  double currentUctConstant = 1.5;

  int numberOfSimulations;

  public MonteCarloTreeSearch(AdversaryLearningConfiguration configuration) {

    this.currentUctConstant = configuration.getuctConstantFactor();
    this.numberOfSimulations = configuration.getNumberOfMonteCarloSimulations();
  }

  void playout(TreeNode treeNode, Game game, ComputationGraph computationGraph) {

    while (treeNode.isExpanded()) {

      treeNode = treeNode.selectMove(this.currentUctConstant);
      game.makeMove(treeNode.lastMove, treeNode.lastMoveColor);
    }

    if (game.gameEnded()) {

      double endResult = game.getEndResult(treeNode.lastMoveColor);

      if (Game.MIN_PLAYER == treeNode.lastMoveColor) {

        endResult = 1 - endResult;
      }

      treeNode.updateRecursiv(endResult);

    } else {

      INDArray currentBoard = game.getCurrentBoard();

      long[] newShape = new long[currentBoard.shape().length + 1];
      System.arraycopy(currentBoard.shape(), 0, newShape, 1, currentBoard.shape().length);
      newShape[0] = 1;
      INDArray oneBatchBoard = currentBoard.reshape(newShape);
      INDArray[] neuralNetOutput = computationGraph.output(oneBatchBoard);

      INDArray actionProbabilities = neuralNetOutput[0];
      double leafValue = neuralNetOutput[1].getDouble(0);

      INDArray validMovesMask = game.getValidMoves();
      INDArray validActionProbabilities = actionProbabilities.mul(validMovesMask);

      Number validActionsSum = validActionProbabilities.sumNumber();

      if (validActionsSum.doubleValue() > 0) {
        validActionProbabilities = validActionProbabilities.div(validActionsSum);
      } else {
        validActionProbabilities = validMovesMask.div(validMovesMask.sumNumber());
      }
      treeNode.updateRecursiv(1 - leafValue);

      treeNode.expand(game, validActionProbabilities);
    }
  }

  public INDArray getActionValues(Game currentGame, double temperature, ComputationGraph computationGraph) {

    TreeNode treeNode = new TreeNode(-1, currentGame.getOtherPlayer(currentGame.getCurrentPlayer()), 0, 1.0, 0.5, null);

    return this.getActionValues(currentGame, treeNode, temperature, computationGraph);
  }

  public INDArray getActionValues(Game currentGame, TreeNode treeNode, double temperature, ComputationGraph computationGraph) {

    int playouts = 0;

    while (playouts < numberOfSimulations) {

      Game newGameInstance = currentGame.createNewInstance();
      this.playout(treeNode, newGameInstance, computationGraph);
      playouts++;
    }

    int[] visitedCounts = new int[currentGame.getNumberOfCurrentMoves()];
    int maxVisitedCounts = 0;

    for (int index = 0; index < currentGame.getNumberOfCurrentMoves(); index++) {

      if (treeNode.containsChildMoveIndex(index)) {

        visitedCounts[index] = treeNode.getChildWithMoveIndex(index).timesVisited;
        if (visitedCounts[index] > maxVisitedCounts) {

          maxVisitedCounts = visitedCounts[index];
        }
      }
    }

    INDArray moveProbabilities = Nd4j.zeros(currentGame.getNumberOfCurrentMoves());

    if (0 == temperature) {

      INDArray visitedCountsArray = Nd4j.createFromArray(visitedCounts);

      INDArray visitedCountsMaximums = Nd4j.where(visitedCountsArray.gte(visitedCountsArray.amax(0).getNumber(0)), null,
          null)[0];

      visitedCountsArray.close();

      moveProbabilities.putScalar(
          visitedCountsMaximums
              .getInt(AdversaryLearningConstants.randomGenerator.nextInt((int) visitedCountsMaximums.length())),
          AdversaryLearningConstants.ONE);

      return moveProbabilities;
    }

    INDArray softmaxParameters = Nd4j.zeros(currentGame.getNumberOfCurrentMoves());
    Set<Integer> validMoveIndices = currentGame.getValidMoveIndices();
    for (int index : validMoveIndices) {

      softmaxParameters.putScalar(index, (1 / temperature) * Math.log(visitedCounts[index] + 1e-8));
    }

    double maxSoftmaxParameter = softmaxParameters.maxNumber().doubleValue();

    for (int index : validMoveIndices) {

      double softmaxParameter = softmaxParameters.getDouble(index);
      if (logger.isWarnEnabled() && Double.isInfinite(softmaxParameter)) {
        logger.warn(
            "Infinite softmax param {} for valid move index {}, NaN results expected."
                + " Is {} and getValidMoveIndices() OK ?" + "getValidMoveIndices() had {}, but visitedCounts {}",
            softmaxParameter, index, currentGame, Arrays.toString(visitedCounts));
      }

      moveProbabilities.putScalar(index, Math.exp(softmaxParameter - maxSoftmaxParameter));
    }

    moveProbabilities = moveProbabilities.div(moveProbabilities.sumNumber());

    return moveProbabilities;
  }
}
