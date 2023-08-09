Alpha Zero learning Tic Tac Toe
===============================

This submodule is an example implementation using the general other module [alpha-zero-adversary-learning](../alpha-zero-adversary-learning).

## Implementation details

### Residual net architecture

```
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
```

### Adversary learning configuration

```
 learningRate: MapSchedule(scheduleType=ITERATION, values={0=0.001}, allKeysSorted=[0])
 batch size: 8192
 dirichletAlpha: 0.8
 dirichletWeight: 0.4
 numberOfAllAvailableMoves: 9
 numberOfEpisodesBeforePotentialUpdate: 10
 numberOfEpisodeThreads: 16
 continueTraining: false
 initialIteration: 1
 numberOfIterations: 300
 checkPointIterationsFrequency: 50
 fromNumberOfIterationsReducedTemperature: -1
 fromNumberOfMovesReducedTemperature: -1
 reducedTemperature: 0.0
 maxTrainExamplesHistory: 5000
 maxTrainExamplesHistoryFromIteration: 0
 currentMaxTrainExamplesHistory: 5000
 cpUct: 1.5
 numberOfMonteCarloSimulations: 25
 modelFileName: /home/evolutionsoft/git/alpha-zero-learning/tic-tac-toe/model.bin
 trainExamplesFileNames: /home/evolutionsoft/git/alpha-zero-learning/tic-tac-toe/trainExamples.obj
```

## Learning performance

With the above configuration 300 iterations with 10 self play episodes take around 12 minutes on AMD Ryzen 3950 with an NVIDIA RTX A6000 and CUDA 11.4 support. After those 300 iterations ~4300 of 4520 play through examples from [supervised learning](https://github.com/evolutionsoftswiss/dl4j) are generated.

With 300 iterations the trained model holds the draw with any or almost any of the nine opening moves playing as first or as second player. Additionally enabling a small monte carlo search can make the model stronger.

## Additional classes with main methods
For training only TicTacToeReinforcementLearningMain.java is necessary. Here two additional classes with main methods are provided. They help interpreting the progress of the performed alpha zero training.

There is no possibility to play against the alpha zero trained model as human yet.

For examples and hints about executing the main methods see the parent module [README.md](../README.md#running-the-tic-tac-toe-implementation)

### EvaluationMain.java
With Tic Tac Toe as a special case, because the simplicity allows a full minimax search, there are generated labels from the [supervised dl4j learning project](https://github.com/evolutionsoftswiss/dl4j). This evaluation is done after each iteration and shows also a learning progress. Alpha zero learns some same moves, but probably also find different correct moves.

```
========================Evaluation Metrics========================
 # of classes:    9
 Accuracy:        0.6659
 Precision:       0.6901
 Recall:          0.7667
 F1 Score:        0.6974
Precision, recall & F1: macro-averaged (equally weighted avg. of 9 classes)


=========================Confusion Matrix=========================
   0   1   2   3   4   5   6   7   8
-------------------------------------
 531  66 128  52 303  74 125  67 151 | 0 = 0
   2 228   7   6  53   6  19   2  16 | 1 = 1
   3   2 384  13  72   8  14  16  14 | 2 = 2
   1   1   0 208  51   5   4   4  15 | 3 = 3
   9   2   4   0 586   5  17  11  20 | 4 = 4
   1   0   0   1  12 210  10   8   2 | 5 = 5
   1   3   6   0  24   3 340   5  13 | 6 = 6
   4   0   2   1  15   0   1 190   2 | 7 = 7
   1   2   3   1  18   1   2   0 333 | 8 = 8

Confusion matrix format: Actual (rowClass) predicted as (columnClass) N times
==================================================================
```

### TicTacToeGamesMain.java
Play 3*9 games with all possible opening moves as X and O player, a total of 54 games. As a more difficult test than with monte carlo search, use neural net output only.
