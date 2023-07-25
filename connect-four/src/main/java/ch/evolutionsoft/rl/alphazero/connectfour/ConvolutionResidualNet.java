package ch.evolutionsoft.rl.alphazero.connectfour;

import static ch.evolutionsoft.rl.alphazero.ConvolutionalResidualNetConstants.*;

import org.deeplearning4j.nn.conf.CNN2DFormat;
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
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer.AlgoMode;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.nd4j.linalg.schedule.ISchedule;

import ch.evolutionsoft.rl.AdversaryLearningConstants;

public class ConvolutionResidualNet {

  private static final String BLOCK6_SEPCONV2_ACT = "block6_sepconv2_act";
  private ISchedule learningRateSchedule;

  public ConvolutionResidualNet(ISchedule learningRateSchedule) {

    this.learningRateSchedule = learningRateSchedule;
  }

  public NeuralNetConfiguration.Builder createGeneralConfiguration() {
    
    return new NeuralNetConfiguration.Builder().
        seed(AdversaryLearningConstants.DEFAULT_SEED).
        updater(new Adam(learningRateSchedule)).
        weightDecay(1e-5, false).
        convolutionMode(ConvolutionMode.Strict).
        weightInit(WeightInit.RELU).
        cudnnAlgoMode(AlgoMode.NO_WORKSPACE);
  }

  public ComputationGraphConfiguration createConvolutionalGraphConfiguration() {

    ComputationGraphConfiguration.GraphBuilder graphBuilder = new ComputationGraphConfiguration.GraphBuilder(
        createGeneralConfiguration()).addInputs(INPUT).setInputTypes(InputType.convolutional(6, 7, 3, CNN2DFormat.NCHW))
            // block1
            .addLayer(BLOCK1_CONVOLUTION1,
                new ConvolutionLayer.Builder(2, 2).stride(1, 1).nIn(3).nOut(16).hasBias(false).build(), INPUT)
            .addLayer(BLOCK1_CONV1_BATCH_NORMALIZATION, new BatchNormalization(), BLOCK1_CONVOLUTION1)
            .addLayer(BLOCK1_CONVOLUTION1_ACTIVATION, new ActivationLayer(Activation.LEAKYRELU),
                BLOCK1_CONV1_BATCH_NORMALIZATION)
            .addLayer(BLOCK1_CONVOLUTION2,
                new ConvolutionLayer.Builder(2, 2).stride(1, 1).nOut(32).hasBias(false).build(),
                BLOCK1_CONVOLUTION1_ACTIVATION)
            .addLayer(BLOCK1_CONVOLUTION2_BATCH_NORMALIZATION, new BatchNormalization(), BLOCK1_CONVOLUTION2)
            .addLayer(BLOCK1_CONV2_ACTIVATION, new ActivationLayer(Activation.LEAKYRELU),
                BLOCK1_CONVOLUTION2_BATCH_NORMALIZATION)

            // residual1
            .addLayer(RESIDUAL1_CONVOLUTION,
                new ConvolutionLayer.Builder(1, 1).stride(1, 1).nOut(32).hasBias(false)
                    .convolutionMode(ConvolutionMode.Same).build(),
                BLOCK1_CONV2_ACTIVATION)
            .addLayer(RESIDUAL1_BATCH_NORMALIZATION, new BatchNormalization(), RESIDUAL1_CONVOLUTION)

            // block2
            .addLayer(BLOCK2_SEPARABLE_CONVOLUTION1,
                new SeparableConvolution2D.Builder(2, 2).nOut(32).hasBias(false).convolutionMode(ConvolutionMode.Same)
                    .build(),
                BLOCK1_CONV2_ACTIVATION)
            .addLayer(BLOCK2_SEPARABLE_CONVOLUTION1_BATCH_NORMALIZATION, new BatchNormalization(),
                BLOCK2_SEPARABLE_CONVOLUTION1)
            .addLayer(BLOCK2_SEPCONV1_ACTIVATION, new ActivationLayer(Activation.LEAKYRELU),
                BLOCK2_SEPARABLE_CONVOLUTION1_BATCH_NORMALIZATION)
            .addLayer(BLOCK2_SEPARABLE_CONVOLUTION2,
                new SeparableConvolution2D.Builder(2, 2).nOut(32).hasBias(false).convolutionMode(ConvolutionMode.Same)
                    .build(),
                BLOCK2_SEPCONV1_ACTIVATION)
            .addLayer(BLOCK2_SEPARABLE_CONVOLUTION2_BATCH_NORMALIZATION, new BatchNormalization(),
                BLOCK2_SEPARABLE_CONVOLUTION2)

            .addVertex(ADD1, new ElementWiseVertex(ElementWiseVertex.Op.Add), BLOCK2_SEPARABLE_CONVOLUTION2_BATCH_NORMALIZATION, RESIDUAL1_BATCH_NORMALIZATION)

            // residual2
            .addLayer("residual2_conv",
                new ConvolutionLayer.Builder(1, 1).stride(1, 1).nOut(64).hasBias(false)
                    .convolutionMode(ConvolutionMode.Same).build(),
                "add1")
            .addLayer("residual2_bn", new BatchNormalization(), "residual2_conv")

            // block3
            .addLayer("block3_sepconv1_act", new ActivationLayer(Activation.LEAKYRELU), "add1")
            .addLayer("block3_sepconv1",
                new SeparableConvolution2D.Builder(2, 2).nOut(64).hasBias(false).convolutionMode(ConvolutionMode.Same)
                    .build(),
                "block3_sepconv1_act")
            .addLayer("block3_sepconv1_bn", new BatchNormalization(), "block3_sepconv1")
            .addLayer("block3_sepconv2_act", new ActivationLayer(Activation.LEAKYRELU), "block3_sepconv1_bn")
            .addLayer("block3_sepconv2",
                new SeparableConvolution2D.Builder(2, 2).nOut(64).hasBias(false).convolutionMode(ConvolutionMode.Same)
                    .build(),
                "block3_sepconv2_act")
            .addLayer("block3_sepconv2_bn", new BatchNormalization(), "block3_sepconv2")
            .addVertex("add2", new ElementWiseVertex(ElementWiseVertex.Op.Add), "block3_sepconv2_bn", "residual2_bn")

            // residual3
            .addLayer("residual3_conv",
                new ConvolutionLayer.Builder(1, 1).stride(1, 1).nOut(128).hasBias(false)
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
            .addVertex("add3", new ElementWiseVertex(ElementWiseVertex.Op.Add), "block4_sepconv2_bn", "residual3")

            // residual4
            /*.addLayer("residual4_conv",
                new ConvolutionLayer.Builder(1, 1).stride(1, 1).nOut(128).hasBias(false)
                    .convolutionMode(ConvolutionMode.Same).build(),
                "add3")
            .addLayer("residual4", new BatchNormalization(), "residual4_conv")

            // block5
            .addLayer("block5_sepconv1_act", new ActivationLayer(Activation.LEAKYRELU), "add3")
            .addLayer("block5_sepconv1",
                new SeparableConvolution2D.Builder(2, 2).nOut(128).hasBias(false).convolutionMode(ConvolutionMode.Same)
                    .build(),
                "block5_sepconv1_act")
            .addLayer("block5_sepconv1_bn", new BatchNormalization(), "block5_sepconv1")
            .addLayer("block5_sepconv2_act", new ActivationLayer(Activation.LEAKYRELU), "block5_sepconv1_bn")
            .addLayer("block5_sepconv2",
                new SeparableConvolution2D.Builder(2, 2).nOut(128).hasBias(false).convolutionMode(ConvolutionMode.Same)
                    .build(),
                "block5_sepconv2_act")
            .addLayer("block5_sepconv2_bn", new BatchNormalization(), "block5_sepconv2")
            .addVertex("add4", new ElementWiseVertex(ElementWiseVertex.Op.Add), "block5_sepconv2_bn", "residual4")*/

            // block5
            .addLayer("block6_sepconv1_act", new ActivationLayer(Activation.LEAKYRELU), "add3")
            .addLayer("block6_sepconv1",
                new SeparableConvolution2D.Builder(2, 2).nOut(128).hasBias(false).convolutionMode(ConvolutionMode.Same)
                    .build(),
                "block6_sepconv1_act")
            .addLayer("block6_sepconv1_bn", new BatchNormalization(), "block6_sepconv1")
            .addLayer(BLOCK6_SEPCONV2_ACT, new ActivationLayer(Activation.LEAKYRELU), "block6_sepconv1_bn");
            
            graphBuilder
            
            .addLayer("dense1", new DenseLayer.Builder().
                nOut(128).
                activation(Activation.LEAKYRELU).
                build(), BLOCK6_SEPCONV2_ACT)
            
            .addLayer("dense2", new DenseLayer.Builder().
                nOut(64).
                activation(Activation.LEAKYRELU).
                build(), BLOCK6_SEPCONV2_ACT)
            
            .addLayer(AdversaryLearningConstants.DEFAULT_OUTPUT_LAYER_NAME, new OutputLayer
                .Builder(LossFunction.MCXENT)
                .nOut(7)
                .activation(Activation.SOFTMAX)
                .build(), "dense1")
            
            .addLayer(AdversaryLearningConstants.DEFAULT_OUTPUT_LAYER_NAME + "_value", new OutputLayer
                .Builder(LossFunction.MSE)
                .nOut(1)
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .build(), "dense2")

        .setOutputs(AdversaryLearningConstants.DEFAULT_OUTPUT_LAYER_NAME, AdversaryLearningConstants.DEFAULT_OUTPUT_LAYER_NAME + "_value");
    
    return graphBuilder.build();
  }

}
