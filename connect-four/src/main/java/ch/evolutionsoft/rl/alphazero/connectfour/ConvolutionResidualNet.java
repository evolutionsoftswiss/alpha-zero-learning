package ch.evolutionsoft.rl.alphazero.connectfour;

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

  public ConvolutionResidualNet(double learningRate) {

    this.learningRate = learningRate;
  }

  public ConvolutionResidualNet(ISchedule learningRateSchedule) {

    this.learningRateSchedule = learningRateSchedule;
  }

  public NeuralNetConfiguration.Builder createGeneralConfiguration() {

    if (null != this.learningRateSchedule) {

      return new NeuralNetConfiguration.Builder().
          seed(AdversaryLearningConstants.DEFAULT_SEED).
          updater(new Adam(learningRateSchedule)).
          convolutionMode(ConvolutionMode.Strict).
          weightInit(WeightInit.RELU);       
    }
    
    return new NeuralNetConfiguration.Builder().
        seed(AdversaryLearningConstants.DEFAULT_SEED).
        updater(new Adam(learningRate)).
        convolutionMode(ConvolutionMode.Strict).
        weightInit(WeightInit.RELU);
  }

  public ComputationGraphConfiguration createConvolutionalGraphConfiguration() {

    ComputationGraphConfiguration.GraphBuilder graphBuilder = new ComputationGraphConfiguration.GraphBuilder(
        createGeneralConfiguration()).addInputs(INPUT).setInputTypes(InputType.convolutional(3, 6, 7))
            // block1
            .addLayer(BLOCK1_CONVOLUTION1,
                new ConvolutionLayer.Builder(2, 2).stride(1, 1).nIn(3).nOut(8).hasBias(false).build(), INPUT)
            .addLayer(BLOCK1_CONV1_BATCH_NORMALIZATION, new BatchNormalization(), BLOCK1_CONVOLUTION1)
            .addLayer(BLOCK1_CONVOLUTION1_ACTIVATION, new ActivationLayer(Activation.LEAKYRELU),
                BLOCK1_CONV1_BATCH_NORMALIZATION)
            .addLayer(BLOCK1_CONVOLUTION2,
                new ConvolutionLayer.Builder(2, 2).stride(1, 1).padding(1, 1).nOut(16).hasBias(false).build(),
                BLOCK1_CONVOLUTION1_ACTIVATION)
            .addLayer(BLOCK1_CONVOLUTION2_BATCH_NORMALIZATION, new BatchNormalization(), BLOCK1_CONVOLUTION2)
            .addLayer(BLOCK1_CONV2_ACTIVATION, new ActivationLayer(Activation.LEAKYRELU),
                BLOCK1_CONVOLUTION2_BATCH_NORMALIZATION)

            // residual1
            .addLayer(RESIDUAL1_CONVOLUTION,
                new ConvolutionLayer.Builder(2, 2).stride(1, 1).nOut(16).hasBias(false)
                    .convolutionMode(ConvolutionMode.Same).build(),
                BLOCK1_CONV2_ACTIVATION)
            .addLayer(RESIDUAL1, new BatchNormalization(), RESIDUAL1_CONVOLUTION)

            // block2
            .addLayer(BLOCK2_SEPARABLE_CONVOLUTION1,
                new SeparableConvolution2D.Builder(2, 2).nOut(16).hasBias(false).convolutionMode(ConvolutionMode.Same)
                    .build(),
                BLOCK1_CONV2_ACTIVATION)
            .addLayer(BLOCK2_SEPARABLE_CONVOLUTION1_BATCH_NORMALIZATION, new BatchNormalization(),
                BLOCK2_SEPARABLE_CONVOLUTION1)
            .addLayer(BLOCK2_SEPCONV1_ACTIVATION, new ActivationLayer(Activation.LEAKYRELU),
                BLOCK2_SEPARABLE_CONVOLUTION1_BATCH_NORMALIZATION)
            .addLayer(BLOCK2_SEPARABLE_CONVOLUTION2,
                new SeparableConvolution2D.Builder(2, 2).nOut(16).hasBias(false).convolutionMode(ConvolutionMode.Same)
                    .build(),
                BLOCK2_SEPCONV1_ACTIVATION)
            .addLayer(BLOCK2_SEPARABLE_CONVOLUTION2_BATCH_NORNMALIZATION, new BatchNormalization(),
                BLOCK2_SEPARABLE_CONVOLUTION2)
            .addLayer(BLOCK2_POOL,
                new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG).kernelSize(2, 2).stride(1, 1)
                    .convolutionMode(ConvolutionMode.Same).build(),
                BLOCK2_SEPARABLE_CONVOLUTION2_BATCH_NORNMALIZATION)

            .addVertex(ADD1, new ElementWiseVertex(ElementWiseVertex.Op.Add), BLOCK2_POOL, RESIDUAL1)

            // residual2
            .addLayer("residual2_conv",
                new ConvolutionLayer.Builder(2, 2).stride(1, 1).nOut(32).hasBias(false)
                    .convolutionMode(ConvolutionMode.Same).build(),
                "add1")
            .addLayer("residual2", new BatchNormalization(), "residual2_conv")

            // block3
            .addLayer("block3_sepconv1_act", new ActivationLayer(Activation.LEAKYRELU), "add1")
            .addLayer("block3_sepconv1",
                new SeparableConvolution2D.Builder(2, 2).nOut(32).hasBias(false).convolutionMode(ConvolutionMode.Same)
                    .build(),
                "block3_sepconv1_act")
            .addLayer("block3_sepconv1_bn", new BatchNormalization(), "block3_sepconv1")
            .addLayer("block3_sepconv2_act", new ActivationLayer(Activation.LEAKYRELU), "block3_sepconv1_bn")
            .addLayer("block3_sepconv2",
                new SeparableConvolution2D.Builder(2, 2).nOut(32).hasBias(false).convolutionMode(ConvolutionMode.Same)
                    .build(),
                "block3_sepconv2_act")
            .addLayer("block3_sepconv2_bn", new BatchNormalization(), "block3_sepconv2")
            .addLayer("block3_pool",
                new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(1, 1)
                    .convolutionMode(ConvolutionMode.Same).build(),
                "block3_sepconv2_bn")
            .addVertex("add2", new ElementWiseVertex(ElementWiseVertex.Op.Add), "block3_pool", "residual2")

            // residual3
            .addLayer("residual3_conv",
                new ConvolutionLayer.Builder(1, 1).stride(2, 2).nOut(128).hasBias(false)
                    .convolutionMode(ConvolutionMode.Same).build(),
                "add2")
            .addLayer("residual3", new BatchNormalization(), "residual3_conv")

            // block4
            .addLayer("block4_sepconv1_act", new ActivationLayer(Activation.LEAKYRELU), "add2")
            .addLayer("block4_sepconv1",
                new SeparableConvolution2D.Builder(2, 2).nOut(128).hasBias(false).convolutionMode(ConvolutionMode.Same)
                    .build(),
                "block4_sepconv1_act")
            .addLayer("block4_sepconv1_bn", new BatchNormalization(), "block4_sepconv1")
            .addLayer("block4_sepconv2_act", new ActivationLayer(Activation.LEAKYRELU), "block4_sepconv1_bn")
            .addLayer("block4_sepconv2",
                new SeparableConvolution2D.Builder(2, 2).nOut(128).hasBias(false).convolutionMode(ConvolutionMode.Same)
                    .build(),
                "block4_sepconv2_act")
            .addLayer("block4_sepconv2_bn", new BatchNormalization(), "block4_sepconv2")
            .addLayer("block4_pool",
                new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2)
                    .convolutionMode(ConvolutionMode.Same).build(),
                "block4_sepconv2_bn")
            .addVertex("add3", new ElementWiseVertex(ElementWiseVertex.Op.Add), "block4_pool", "residual3")

            .addLayer("policy_conv",
                new SeparableConvolution2D.Builder(1, 1).nOut(8).hasBias(false).convolutionMode(ConvolutionMode.Same)
                .build(), "add3")
            
            .addLayer("dense1", new DenseLayer.Builder().
                nOut(64).
                activation(Activation.LEAKYRELU).
                build(), "policy_conv")

            .addLayer("value_conv",
                new SeparableConvolution2D.Builder(1, 1).nOut(2).hasBias(false).convolutionMode(ConvolutionMode.Same)
                .build(), "add3")
            
            .addLayer("dense2", new DenseLayer.Builder().
                nOut(32).
                activation(Activation.LEAKYRELU).
                build(), "value_conv")
            
            .addLayer(AdversaryLearningConstants.DEFAULT_OUTPUT_LAYER_NAME, new OutputLayer.Builder()
                .nOut(7)
                .activation(Activation.SOFTMAX)
                .build(), "dense1")
            
            .addLayer(AdversaryLearningConstants.DEFAULT_OUTPUT_LAYER_NAME + "_value", new OutputLayer.Builder()
                .nOut(1)
                .activation(Activation.SIGMOID)
                .lossFunction(LossFunction.MSE)
                .build(), "dense2")

        .setOutputs(AdversaryLearningConstants.DEFAULT_OUTPUT_LAYER_NAME, AdversaryLearningConstants.DEFAULT_OUTPUT_LAYER_NAME + "_value");
    
    return graphBuilder.build();
  }

}