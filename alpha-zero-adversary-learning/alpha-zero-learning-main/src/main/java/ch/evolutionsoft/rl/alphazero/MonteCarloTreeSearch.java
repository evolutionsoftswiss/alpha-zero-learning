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

  void playout(TreeNode rootNode, Game game, ComputationGraph computationGraph) {

    TreeNode mctsRoot = rootNode;
    TreeNode currentNode = rootNode;

    while (currentNode.isExpanded()) {
      currentNode = currentNode.selectMove(this.currentUctConstant);
      game.makeMove(currentNode.lastMove, currentNode.lastMoveColor);
    }
    
    if (game.gameEnded()) {

      double endResult = game.getEndResult(currentNode.lastMoveColor);

      if (Game.MIN_PLAYER == currentNode.lastMoveColor) {

        endResult = 1 - endResult;
      }
      currentNode.updateRecursiv(endResult, mctsRoot);
      
    } else {
      INDArray currentBoard = game.getCurrentBoard();

      long[] newShape = new long[currentBoard.shape().length + 1];
      System.arraycopy(currentBoard.shape(), 0, newShape, 1, currentBoard.shape().length);
      newShape[0] = 1;
      
      INDArray oneBatchBoard = currentBoard.reshape(newShape);
      INDArray[] neuralNetOutput; neuralNetOutput = computationGraph.output(oneBatchBoard);
      
      
      INDArray actionProbabilities = neuralNetOutput[0];
      double leafValue = neuralNetOutput[1].getDouble(0);
      
      INDArray validMovesMask = game.getValidMoves();
      INDArray validActionProbabilities = actionProbabilities.mul(validMovesMask);
      
      Number validActionsSum = validActionProbabilities.sumNumber();
      INDArray normalizedValidActionProbabilities;

      if (validActionsSum.doubleValue() > 0) {
        normalizedValidActionProbabilities = validActionProbabilities.div(validActionsSum);
      } else {
        normalizedValidActionProbabilities = validMovesMask.div(validMovesMask.sumNumber());
      }
      currentNode.updateRecursiv(1 - leafValue, mctsRoot);
      currentNode.expand(game, normalizedValidActionProbabilities);
      }
  }

  public INDArray getActionValues(Game currentGame, double temperature, ComputationGraph computationGraph) {

    TreeNode treeNode = new TreeNode(-1, currentGame.getOtherPlayer(currentGame.getCurrentPlayer()), 0, 1.0, 0.5, null);

    return this.getActionValues(currentGame, treeNode, temperature, computationGraph);
  }

  public INDArray getActionValues(Game currentGame, TreeNode rootNode, double temperature, ComputationGraph computationGraph) {

    int playouts = 0;

    while (playouts < numberOfSimulations) {

      Game newGameInstance = currentGame.createNewInstance();
      
      this.playout(rootNode, newGameInstance, computationGraph);
      
      playouts++;
    }

    int[] visitedCounts = new int[currentGame.getNumberOfCurrentMoves()];
    int maxVisitedCounts = 0;

    for (int index = 0; index < currentGame.getNumberOfCurrentMoves(); index++) {

      if (rootNode.containsChildMoveIndex(index)) {

        visitedCounts[index] = rootNode.getChildWithMoveIndex(index).timesVisited;
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

  public TreeNode updateMonteCarloSearchRoot(Game game, TreeNode lastRoot, int moveAction) {

    if (lastRoot.containsChildMoveIndex(moveAction)) {

      return lastRoot.getChildWithMoveIndex(moveAction);
    } else {

      logger.warn("{}", game);
      logger.error("no child with move {} " + "found for current root node with last move {}", moveAction,
          lastRoot.lastMove);

      return null;
    }
  }
}
