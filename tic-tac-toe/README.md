Alpha Zero learning TicTacToe
=============================

This submodule is an example implementation using the general other module [alpha-zero-adversary-learning](../alpha-zero-learning/tree/master/alpha-zero-adversary-learning).

## Implementation details

### Residual net architecture
	=============================================================================================================================================================
	VertexName (VertexType)                         nIn,nOut   TotalParams   ParamsShape                                                 Vertex Inputs           
	=============================================================================================================================================================
	input (InputVertex)                             -,-        -             -                                                           -                       
	block1_conv1 (ConvolutionLayer)                 3,7        84            W:{7,3,2,2}                                                 [input]                 
	block1_conv1_bn (BatchNormalization)            7,7        28            gamma:{1,7}, beta:{1,7}, mean:{1,7}, log10stdev:{1,7}       [block1_conv1]          
	block1_conv1_act (ActivationLayer)              -,-        0             -                                                           [block1_conv1_bn]       
	block1_conv2 (ConvolutionLayer)                 7,14       392           W:{14,7,2,2}                                                [block1_conv1_act]      
	block1_conv2_bn (BatchNormalization)            14,14      56            gamma:{1,14}, beta:{1,14}, mean:{1,14}, log10stdev:{1,14}   [block1_conv2]          
	block1_conv2_act (ActivationLayer)              -,-        0             -                                                           [block1_conv2_bn]       
	residual1_conv (ConvolutionLayer)               14,14      784           W:{14,14,2,2}                                               [block1_conv2_act]      
	block2_sepconv1 (SeparableConvolution2DLayer)   14,14      252           W:{1,14,2,2}, pW:{14,14,1,1}                                [block1_conv2_act]      
	residual1 (BatchNormalization)                  14,14      56            gamma:{1,14}, beta:{1,14}, mean:{1,14}, log10stdev:{1,14}   [residual1_conv]        
	block2_sepconv1_bn (BatchNormalization)         14,14      56            gamma:{1,14}, beta:{1,14}, mean:{1,14}, log10stdev:{1,14}   [block2_sepconv1]       
	block2_sepconv1_act (ActivationLayer)           -,-        0             -                                                           [block2_sepconv1_bn]    
	block2_sepconv2 (SeparableConvolution2DLayer)   14,14      252           W:{1,14,2,2}, pW:{14,14,1,1}                                [block2_sepconv1_act]   
	block2_sepconv2_bn (BatchNormalization)         14,14      56            gamma:{1,14}, beta:{1,14}, mean:{1,14}, log10stdev:{1,14}   [block2_sepconv2]       
	block2_pool (SubsamplingLayer)                  -,-        0             -                                                           [block2_sepconv2_bn]    
	add1 (ElementWiseVertex)                        -,-        -             -                                                           [block2_pool, residual1]
	policy_conv (SeparableConvolution2DLayer)       14,8       126           W:{1,14,1,1}, pW:{8,14,1,1}                                 [add1]                  
	value_conv (SeparableConvolution2DLayer)        14,2       42            W:{1,14,1,1}, pW:{2,14,1,1}                                 [add1]                  
	dense1 (DenseLayer)                             72,32      2,336         W:{72,32}, b:{1,32}                                         [policy_conv]           
	dense2 (DenseLayer)                             18,16      304           W:{18,16}, b:{1,16}                                         [value_conv]            
	OutputLayer (OutputLayer)                       32,9       297           W:{32,9}, b:{1,9}                                           [dense1]                
	OutputLayer_value (OutputLayer)                 16,1       17            W:{16,1}, b:{1,1}                                           [dense2]                
	-------------------------------------------------------------------------------------------------------------------------------------------------------------
	            Total Parameters:  5,138
	        Trainable Parameters:  5,138
	           Frozen Parameters:  0
	=============================================================================================================================================================

### Adversary learning configuration
	 learningRate: MapSchedule(scheduleType=ITERATION, values={0=0.002, 200=0.001}, allKeysSorted=[0, 200])
	 batch size: 8192
	 dirichletAlpha: 1.1
	 dirichletWeight: 0.45
	 alwaysUpdateNeuralNetwork: true
	 gamesToGetNewNetworkWinRatio: -
	 gamesWinRatioThresholdNewNetworkUpdate: -
	 numberOfEpisodesBeforePotentialUpdate: 10
	 iterationStart: 1
	 numberOfIterations: 250
	 checkPointIterationsFrequency: 50
	 fromNumberOfIterationsTemperatureZero: -1
	 fromNumberOfMovesTemperatureZero: 3
	 maxTrainExamplesHistory: 5000
	 cpUct: 0.8
	 numberOfMonteCarloSimulations: 30
	 bestModelFileName: /home/evolutionsoft/git/alpha-zero-learning/tic-tac-toe/bestmodel.bin
	 trainExamplesFileName: /home/evolutionsoft/git/alpha-zero-learning/tic-tac-toe/trainExamples.obj

## Learning performance

With the above configuration 250 iterations with 10 self play episodes take around 45 minutes on i7-5700 with avx2 enabled build. After those 250 iterations ~4000 of 4520 play through examples from [supervised learning](https://github.com/evolutionsoftswiss/dl4j) are generated.

With 250 iterations the trained model holds the draw with any of the nine opening moves playing as first or as second player.

## Additional classes with main methods
For training only TicTacToeReinforcementLearningMain.java is necessary. Here two additional classes with main methods are provided. They help interpreting the progress of the performed alpha zero training.

There is no possibility to play against the alpha zero trained model as human yet.

For examples and hints about executing the main methods see the parent module [README.md](../README.md#running-the-tic-tac-toe-implementation)

### EvaluationMain.java
With Tic Tac Toe as a special case, because the simplicity allows a full min max search, there are generated labels from the [supervised dl4j learning project](https://github.com/evolutionsoftswiss/dl4j). This evaluation is done after each iteration and shows also a learning progress. Alpha zero learns some same moves, but probably also find different correct moves.

	========================Evaluation Metrics========================
	 # of classes:    9
	 Accuracy:        0.5478
	 Precision:       0.5617
	 Recall:          0.5999
	 F1 Score:        0.5477
	Precision, recall & F1: macro-averaged (equally weighted avg. of 9 classes)
	
	
	=========================Confusion Matrix=========================
	   0   1   2   3   4   5   6   7   8
	-------------------------------------
	 490  63 122  49 371  67 116  55 116 | 0 = 0
	   8 184  24  25  75  22  37   7  39 | 1 = 1
	  13   1 341  31 112   4  27  21  31 | 2 = 2
	   5   9   5 146  62   3  20  24  39 | 3 = 3
	   7   8  13  10 572  13  24  13  21 | 4 = 4
	   7   6   2   1  20 138  26  19   8 | 5 = 5
	  10   1  12   4  47   8 244   7  27 | 6 = 6
	   5   1   3   5  20   6   5 113  12 | 7 = 7
	   8   4  15   3  33   4   3   0 248 | 8 = 8
	
	Confusion matrix format: Actual (rowClass) predicted as (columnClass) N times
	==================================================================

### TicTacToeGamesMain.java
Play 3*9 games with all possible opening moves as X and O player, a total of 54 games. The model from alpha zero holds always the draw against the perfect TicTacToePerfectResidualNet.bin.

Experience showed that around 50-60% accuracy compared with the generated labels are enough to always reach the draw. There may still be different play through's with other moves than from TicTacToePerfectResidualNet.bin after the first move, where the alpha zero model not has the correct answer learned yet.