Alpha Zero learning Connect Four
================================

This submodule implements Connect Four with general classes from [alpha-zero-adversary-learning](../alpha-zero-adversary-learning).

## Play against the trained model locally
![ConnectFourGUI](https://github.com/evolutionsoftswiss/alpha-zero-learning/blob/feature/connect-four/connect-four/ConnectFourGUI.png)

## Implementation details

### Residual net architecture

```
===========================================================================================================================================================================
VertexName (VertexType)                         nIn,nOut   TotalParams   ParamsShape                                                     Vertex Inputs                     
===========================================================================================================================================================================
input (InputVertex)                             -,-        -             -                                                               -                                 
block1_conv1 (ConvolutionLayer)                 3,16       192           W:{16,3,2,2}                                                    [input]                           
block1_conv1_bn (BatchNormalization)            16,16      64            gamma:{1,16}, beta:{1,16}, mean:{1,16}, log10stdev:{1,16}       [block1_conv1]                    
block1_conv1_act (ActivationLayer)              -,-        0             -                                                               [block1_conv1_bn]                 
block1_conv2 (ConvolutionLayer)                 16,32      2,048         W:{32,16,2,2}                                                   [block1_conv1_act]                
block1_conv2_bn (BatchNormalization)            32,32      128           gamma:{1,32}, beta:{1,32}, mean:{1,32}, log10stdev:{1,32}       [block1_conv2]                    
block1_conv2_act (ActivationLayer)              -,-        0             -                                                               [block1_conv2_bn]                 
residual1_conv (ConvolutionLayer)               32,32      1,024         W:{32,32,1,1}                                                   [block1_conv2_act]                
block2_sepconv1 (SeparableConvolution2DLayer)   32,32      1,152         W:{1,32,2,2}, pW:{32,32,1,1}                                    [block1_conv2_act]                
residual1_bn (BatchNormalization)               32,32      128           gamma:{1,32}, beta:{1,32}, mean:{1,32}, log10stdev:{1,32}       [residual1_conv]                  
block2_sepconv1_bn (BatchNormalization)         32,32      128           gamma:{1,32}, beta:{1,32}, mean:{1,32}, log10stdev:{1,32}       [block2_sepconv1]                 
block2_sepconv1_act (ActivationLayer)           -,-        0             -                                                               [block2_sepconv1_bn]              
block2_sepconv2 (SeparableConvolution2DLayer)   32,32      1,152         W:{1,32,2,2}, pW:{32,32,1,1}                                    [block2_sepconv1_act]             
block2_sepconv2_bn (BatchNormalization)         32,32      128           gamma:{1,32}, beta:{1,32}, mean:{1,32}, log10stdev:{1,32}       [block2_sepconv2]                 
add1 (ElementWiseVertex)                        -,-        -             -                                                               [block2_sepconv2_bn, residual1_bn]
block3_sepconv1_act (ActivationLayer)           -,-        0             -                                                               [add1]                            
residual2_conv (ConvolutionLayer)               32,64      2,048         W:{64,32,1,1}                                                   [add1]                            
block3_sepconv1 (SeparableConvolution2DLayer)   32,64      2,176         W:{1,32,2,2}, pW:{64,32,1,1}                                    [block3_sepconv1_act]             
residual2_bn (BatchNormalization)               64,64      256           gamma:{1,64}, beta:{1,64}, mean:{1,64}, log10stdev:{1,64}       [residual2_conv]                  
block3_sepconv1_bn (BatchNormalization)         64,64      256           gamma:{1,64}, beta:{1,64}, mean:{1,64}, log10stdev:{1,64}       [block3_sepconv1]                 
block3_sepconv2_act (ActivationLayer)           -,-        0             -                                                               [block3_sepconv1_bn]              
block3_sepconv2 (SeparableConvolution2DLayer)   64,64      4,352         W:{1,64,2,2}, pW:{64,64,1,1}                                    [block3_sepconv2_act]             
block3_sepconv2_bn (BatchNormalization)         64,64      256           gamma:{1,64}, beta:{1,64}, mean:{1,64}, log10stdev:{1,64}       [block3_sepconv2]                 
add2 (ElementWiseVertex)                        -,-        -             -                                                               [block3_sepconv2_bn, residual2_bn]
residual3_conv (ConvolutionLayer)               64,128     8,192         W:{128,64,1,1}                                                  [add2]                            
block4_sepconv1_act (ActivationLayer)           -,-        0             -                                                               [add2]                            
residual3 (BatchNormalization)                  128,128    512           gamma:{1,128}, beta:{1,128}, mean:{1,128}, log10stdev:{1,128}   [residual3_conv]                  
block4_sepconv1 (SeparableConvolution2DLayer)   64,128     8,448         W:{1,64,2,2}, pW:{128,64,1,1}                                   [block4_sepconv1_act]             
block4_sepconv1_bn (BatchNormalization)         128,128    512           gamma:{1,128}, beta:{1,128}, mean:{1,128}, log10stdev:{1,128}   [block4_sepconv1]                 
block4_sepconv2_act (ActivationLayer)           -,-        0             -                                                               [block4_sepconv1_bn]              
block4_sepconv2 (SeparableConvolution2DLayer)   128,128    16,896        W:{1,128,2,2}, pW:{128,128,1,1}                                 [block4_sepconv2_act]             
block4_sepconv2_bn (BatchNormalization)         128,128    512           gamma:{1,128}, beta:{1,128}, mean:{1,128}, log10stdev:{1,128}   [block4_sepconv2]                 
add3 (ElementWiseVertex)                        -,-        -             -                                                               [block4_sepconv2_bn, residual3]   
block6_sepconv1_act (ActivationLayer)           -,-        0             -                                                               [add3]                            
block6_sepconv1 (SeparableConvolution2DLayer)   128,128    16,896        W:{1,128,2,2}, pW:{128,128,1,1}                                 [block6_sepconv1_act]             
block6_sepconv1_bn (BatchNormalization)         128,128    512           gamma:{1,128}, beta:{1,128}, mean:{1,128}, log10stdev:{1,128}   [block6_sepconv1]                 
block6_sepconv2_act (ActivationLayer)           -,-        0             -                                                               [block6_sepconv1_bn]              
dense1 (DenseLayer)                             2560,128   327,808       W:{2560,128}, b:{1,128}                                         [block6_sepconv2_act]             
dense2 (DenseLayer)                             2560,64    163,904       W:{2560,64}, b:{1,64}                                           [block6_sepconv2_act]             
OutputLayer (OutputLayer)                       128,7      903           W:{128,7}, b:{1,7}                                              [dense1]                          
OutputLayer_value (OutputLayer)                 64,1       65            W:{64,1}, b:{1,1}                                               [dense2]                          
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            Total Parameters:  560,648
        Trainable Parameters:  560,648
           Frozen Parameters:  0
===========================================================================================================================================================================
```

### Adversary learning configuration

```
 learningRate: MapSchedule(scheduleType=ITERATION, values={0=1.0E-4, 40000=5.0E-5, 100000=1.0E-5}, allKeysSorted=[0, 40000, 100000])
 batch size: 4096
 dirichletAlpha: 0.7
 dirichletWeight: 0.35
 numberOfAllAvailableMoves: 7
 numberOfEpisodesBeforePotentialUpdate: 20
 numberOfEpisodeThreads: 20
 continueTraining: false
 initialIteration: 1
 numberOfIterations: 4000
 checkPointIterationsFrequency: 50
 fromNumberOfIterationsReducedTemperature: -1
 fromNumberOfMovesReducedTemperature: -1
 reducedTemperature: 0.0
 maxTrainExamplesHistory: 81920
 maxTrainExamplesHistoryFromIteration: 300
 currentMaxTrainExamplesHistory: 273
 cpUct: 2.5
 numberOfMonteCarloSimulations: 200
 bestModelFileName: /home/evolutionsoft/git/alpha-zero-learning/connect-four/model.bin
 trainExamplesFileNames: /home/evolutionsoft/git/alpha-zero-learning/connect-four/trainExamples.obj
```
