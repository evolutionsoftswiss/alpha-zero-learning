Alpha Zero adversary learning
=============================

This is the main submodule providing the general parts of Java alpha zero learning.

Alpha zero adversary learning uses 1 for a winning position, 0.5 for a draw and 0 for a losing position.
These differ from the often used 1, 0, -1.

Keep this in mind when using position values or defining the activation function of the value output: TANH does fit a range from -1 to 1, while SIGMOID can be used for 0 to 1.

## Implement your own new board game
You need at least the following to implement your own new game and perform trainings.

1. Provide a subclass of [Game.java](https://github.com/evolutionsoftswiss/alpha-zero-learning/blob/master/alpha-zero-adversary-learning/src/main/java/ch/evolutionsoft/rl/Game.java) like [TicTacToe.java](https://github.com/evolutionsoftswiss/alpha-zero-learning/blob/master/tic-tac-toe/src/main/java/ch/evolutionsoft/rl/tictactoe/TicTacToe.java)
2. Design a ComputationGraph, potentially a convolution residual net, adapted to the game complexity.
3. Provide an adapted [AdversaryLearningConfiguration.java](https://github.com/evolutionsoftswiss/alpha-zero-learning/blob/master/alpha-zero-adversary-learning/src/main/java/ch/evolutionsoft/rl/AdversaryLearningConfiguration.java) to configure parameters of the training.
4. Put 1.-3. together in a main method and let the computer learn the game

Here's the main class of tic-tac-toe giving an overview:

	  public static void main(String[] args) throws IOException {
	    
	    TicTacToeReinforcementLearningMain main = new TicTacToeReinforcementLearningMain();
	    
	    Map<Integer, Double> learningRatesByIterations = new HashMap<>();
	    learningRatesByIterations.put(0, 2e-3);
	    learningRatesByIterations.put(200, 1e-3);
	    MapSchedule learningRateMapSchedule = new MapSchedule(ScheduleType.ITERATION, learningRatesByIterations);
	    AdversaryLearningConfiguration adversaryLearningConfiguration =
	        new AdversaryLearningConfiguration.Builder().
	        learningRateSchedule(learningRateMapSchedule).
	        build();
	   
	    ComputationGraph neuralNet = main.createConvolutionalConfiguration(adversaryLearningConfiguration);
	
	    if (log.isInfoEnabled()) {
	      log.info(neuralNet.summary());
	    }
	    
	    AdversaryLearning adversaryLearning =
	        new AdversaryLearning(
	            new TicTacToe(Game.MAX_PLAYER),
	            neuralNet,
	            adversaryLearningConfiguration);
	    
	    adversaryLearning.performLearning();
	  }
	
	  ComputationGraph createConvolutionalConfiguration(AdversaryLearningConfiguration adversaryLearningConfiguration) {
	
	    ConvolutionResidualNet convolutionalLayerNet =
	        new ConvolutionResidualNet(adversaryLearningConfiguration.getLearningRate());
	
	    if (null != adversaryLearningConfiguration.getLearningRateSchedule()) {
	
	      convolutionalLayerNet =
	          new ConvolutionResidualNet(adversaryLearningConfiguration.getLearningRateSchedule());
	    }
	    
	    ComputationGraphConfiguration convolutionalLayerNetConfiguration =
	        convolutionalLayerNet.createConvolutionalGraphConfiguration();
	
	    ComputationGraph net = new ComputationGraph(convolutionalLayerNetConfiguration);
	    net.init();
	
	    return net;
	  }

### Subclass Game.java
Important hints:
* A concrete Game implementation must keep track of the current board and current player. That means also that you need to update those members after a move.
* You should ensure in the concrete Game implementation that only copies of the member board state are returned. Otherwise you may have unexpected changes to the Game board state.
* A new Game Subclass may use your existing Java classes to manage a board and other variables of the Game, it would then e kind of bridge between your existing board implementation and the alpha zero learning.
* In the case of Tic Tac Toe the whole state about the Game is kept in the Subclass and there is no additional board or Game management necessary

### The ComputationGraph
Deeplearning4j uses ComputationGraph as a generalization of MultiLayerNetwork. With two net outputs for the move probabilities and board value we have to use ComputationGraph instead of MultiLayerNetwork.

Generally you should start with a small ComputationGraph architecture as it consumes less training time. For less complex games it should not be necessary to use a 20 or 40 layer deep architecture like it was used for 19x19 Go.

### The AdversaryLearningConfiguration
It is configurable by a manually implemented Builder. The current defaults are working to learn Tic Tac Toe. For another game you should probable use different values for all available configuration parameters.

See the source for a description of the available training parameters: [AdversaryLearningConfiguration.java](https://github.com/evolutionsoftswiss/alpha-zero-learning/blob/master/alpha-zero-adversary-learning/src/main/java/ch/evolutionsoft/rl/AdversaryLearningConfiguration.java)