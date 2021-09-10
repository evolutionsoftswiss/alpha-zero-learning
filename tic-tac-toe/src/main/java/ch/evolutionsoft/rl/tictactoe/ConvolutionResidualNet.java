package ch.evolutionsoft.rl.tictactoe;

import static ch.evolutionsoft.rl.ConvolutionalResidualNetConstants.*;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ActivationLayer;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SeparableConvolution2D;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.nd4j.linalg.schedule.ISchedule;

import ch.evolutionsoft.rl.AdversaryLearningConstants;

public class ConvolutionResidualNet {
  
  private double learningRate = 1e-3;
  
  private ISchedule learningRateSchedule;
  
  public ConvolutionResidualNet() {
    
  }

  public ConvolutionResidualNet(double learningRate) {

    this.learningRate = learningRate;
  }

  public ConvolutionResidualNet(ISchedule learningRateSchedule) {

    this.learningRateSchedule = learningRateSchedule;
  }
  
  NeuralNetConfiguration.Builder createGeneralConfiguration() {
    
    if (null != this.learningRateSchedule) {

      return new NeuralNetConfiguration.Builder()
          .seed(AdversaryLearningConstants.DEFAULT_SEED)
          .updater(new Adam(learningRateSchedule))
          .convolutionMode(ConvolutionMode.Strict)
          .weightInit(WeightInit.RELU); 
    }

    return new NeuralNetConfiguration.Builder()
        .seed(AdversaryLearningConstants.DEFAULT_SEED)
        .updater(new Adam(learningRate))
        .convolutionMode(ConvolutionMode.Strict)
        .weightInit(WeightInit.RELU);
  }

  public ComputationGraphConfiguration createConvolutionalGraphConfiguration() {

    return new ComputationGraphConfiguration.GraphBuilder(createGeneralConfiguration())
        .addInputs(INPUT).setInputTypes(InputType.convolutional(3, 3, 3))
        // block1
        .addLayer(BLOCK1_CONVOLUTION1,
            new ConvolutionLayer.Builder(2, 2).stride(1, 1).nIn(3).nOut(7).hasBias(false)
                .build(),
            INPUT)
        .addLayer(BLOCK1_CONV1_BATCH_NORMALIZATION, new BatchNormalization(), BLOCK1_CONVOLUTION1)
        .addLayer(BLOCK1_CONVOLUTION1_ACTIVATION, new ActivationLayer(Activation.LEAKYRELU), BLOCK1_CONV1_BATCH_NORMALIZATION)
        .addLayer(BLOCK1_CONVOLUTION2,
            new ConvolutionLayer.Builder(2, 2).stride(1, 1).padding(1, 1).nOut(14).hasBias(false)
                .build(),
            BLOCK1_CONVOLUTION1_ACTIVATION)
        .addLayer(BLOCK1_CONVOLUTION2_BATCH_NORMALIZATION, new BatchNormalization(), BLOCK1_CONVOLUTION2)
        .addLayer(BLOCK1_CONV2_ACTIVATION, new ActivationLayer(Activation.LEAKYRELU), BLOCK1_CONVOLUTION2_BATCH_NORMALIZATION)

        // residual1
        .addLayer(RESIDUAL1_CONVOLUTION,
            new ConvolutionLayer.Builder(2, 2).stride(1, 1).nOut(14).hasBias(false)
                .convolutionMode(ConvolutionMode.Same).build(),
            BLOCK1_CONV2_ACTIVATION)
        .addLayer(RESIDUAL1, new BatchNormalization(), RESIDUAL1_CONVOLUTION)

        // block2
        .addLayer(BLOCK2_SEPARABLE_CONVOLUTION1,
            new SeparableConvolution2D.Builder(2, 2).nOut(14).hasBias(false).convolutionMode(ConvolutionMode.Same)
                .build(),
            BLOCK1_CONV2_ACTIVATION)
        .addLayer(BLOCK2_SEPARABLE_CONVOLUTION1_BATCH_NORMALIZATION, new BatchNormalization(), BLOCK2_SEPARABLE_CONVOLUTION1)
        .addLayer(BLOCK2_SEPCONV1_ACTIVATION, new ActivationLayer(Activation.LEAKYRELU), BLOCK2_SEPARABLE_CONVOLUTION1_BATCH_NORMALIZATION)
        .addLayer(BLOCK2_SEPARABLE_CONVOLUTION2,
            new SeparableConvolution2D.Builder(2, 2).nOut(14).hasBias(false).convolutionMode(ConvolutionMode.Same)
                .build(),
            BLOCK2_SEPCONV1_ACTIVATION)
        .addLayer(BLOCK2_SEPARABLE_CONVOLUTION2_BATCH_NORNMALIZATION, new BatchNormalization(), BLOCK2_SEPARABLE_CONVOLUTION2)
        .addLayer(BLOCK2_POOL,
            new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG).kernelSize(2, 2).stride(1, 1)
                .convolutionMode(ConvolutionMode.Same).build(),
            BLOCK2_SEPARABLE_CONVOLUTION2_BATCH_NORNMALIZATION)
        
        .addVertex(ADD1, new ElementWiseVertex(ElementWiseVertex.Op.Add), BLOCK2_POOL, RESIDUAL1)

        .addLayer("policy_conv",
            new SeparableConvolution2D.Builder(1, 1).nOut(8).hasBias(false).convolutionMode(ConvolutionMode.Same)
            .build(), ADD1)
        
        .addLayer("dense1", new DenseLayer.Builder().
            nOut(32).
            activation(Activation.LEAKYRELU).
            build(), "policy_conv")

        .addLayer("value_conv",
            new SeparableConvolution2D.Builder(1, 1).nOut(2).hasBias(false).convolutionMode(ConvolutionMode.Same)
            .build(), ADD1)
        
        .addLayer("dense2", new DenseLayer.Builder().
            nOut(16).
            activation(Activation.LEAKYRELU).
            build(), "value_conv")
        
        .addLayer(AdversaryLearningConstants.DEFAULT_OUTPUT_LAYER_NAME, new OutputLayer.Builder()
            .nOut(9)
            .activation(Activation.SOFTMAX)
            .build(), "dense1")
        
        .addLayer(AdversaryLearningConstants.DEFAULT_OUTPUT_LAYER_NAME + "_value", new OutputLayer.Builder()
            .nOut(1)
            .activation(Activation.SIGMOID)
            .lossFunction(LossFunction.MSE)
            .build(), "dense2")
 
        .setOutputs(AdversaryLearningConstants.DEFAULT_OUTPUT_LAYER_NAME, AdversaryLearningConstants.DEFAULT_OUTPUT_LAYER_NAME + "_value")

        .build();
  }

}
