package ch.evolutionsoft.rl.tictactoe;

import static ch.evolutionsoft.net.game.NeuralNetConstants.DEFAULT_OUTPUT_LAYER_NAME;
import static ch.evolutionsoft.net.game.NeuralNetConstants.DEFAULT_SEED;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.ConstantDistribution;
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

public class ConvolutionResidualNet {

  private static final String BLOCK2_SEPARABLE_CONVOLUTION1 = "block2_sepconv1";

  private static final String BLOCK2_SEPARABLE_CONVOLUTION1_BATCH_NORMALIZATION = "block2_sepconv1_bn";

  private static final String BLOCK2_SEPARABLE_CONVOLUTION2 = "block2_sepconv2";

  private static final String BLOCK2_SEPARABLE_CONVOLUTION2_BATCH_NORNMALIZATION = "block2_sepconv2_bn";

  private static final String ADD1 = "add1";

  private static final String BLOCK2_POOL = "block2_pool";

  private static final String RESIDUAL1 = "residual1";

  private static final String RESIDUAL1_CONVOLUTION = "residual1_conv";

  private static final String BLOCK1_CONVOLUTION2_BATCH_NORMALIZATION = "block1_conv2_bn";

  private static final String BLOCK1_CONVOLUTION2 = "block1_conv2";

  private static final String BLOCK1_CONVOLUTION1_ACTIVATION = "block1_conv1_act";

  private static final String BLOCK1_CONVOLUTION1 = "block1_conv1";

  private static final String INPUT = "input";

  private static final String BLOCK2_SEPCONV1_ACTIVATION = "block2_sepconv1_act";

  private static final String BLOCK1_CONV2_ACTIVATION = "block1_conv2_act";

  private static final String BLOCK1_CONV1_BATCH_NORMALIZATION = "block1_conv1_bn";

  public static final int CNN_OUTPUT_CHANNELS = 3;
  
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
          .seed(DEFAULT_SEED)
          .updater(new Adam(learningRateSchedule))
          .convolutionMode(ConvolutionMode.Strict)
          .weightInit(WeightInit.RELU); 
    }

    return new NeuralNetConfiguration.Builder()
        .seed(DEFAULT_SEED)
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
        
        .addLayer("dense1", new DenseLayer.Builder().
            nOut(32).
            activation(Activation.LEAKYRELU).
            build(), ADD1)
        
        .addLayer("dense2", new DenseLayer.Builder().
            nOut(16).
            activation(Activation.LEAKYRELU).
            build(), ADD1)
        
        .addLayer(DEFAULT_OUTPUT_LAYER_NAME, new OutputLayer.Builder()
            .nOut(9)
            .activation(Activation.SOFTMAX)
            .weightInit(new ConstantDistribution(0.01))
            .build(), "dense1")
        
        .addLayer(DEFAULT_OUTPUT_LAYER_NAME + "_value", new OutputLayer.Builder()
            .nOut(1)
            .activation(Activation.SIGMOID)
            .weightInit(new ConstantDistribution(0.01))
            .lossFunction(LossFunction.MSE)
            .build(), "dense2")
 
        .setOutputs(DEFAULT_OUTPUT_LAYER_NAME, DEFAULT_OUTPUT_LAYER_NAME + "_value")

        .build();
  }

}
