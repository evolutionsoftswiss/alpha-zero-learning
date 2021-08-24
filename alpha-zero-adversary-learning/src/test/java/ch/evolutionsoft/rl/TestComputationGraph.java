package ch.evolutionsoft.rl;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import ch.evolutionsoft.net.game.NeuralNetConstants;

public class TestComputationGraph extends ComputationGraph {

  public TestComputationGraph() {

    super(new ComputationGraphConfiguration.GraphBuilder(new NeuralNetConfiguration.Builder()).
        addInputs(NeuralNetConstants.DEFAULT_INPUT_LAYER_NAME).setInputTypes(InputType.convolutional(1, 1, 1)).
        addLayer(
            NeuralNetConstants.DEFAULT_OUTPUT_LAYER_NAME,
            new OutputLayer.Builder().activation(Activation.SIGMOID).lossFunction(LossFunction.XENT).nOut(1).build(),
            NeuralNetConstants.DEFAULT_INPUT_LAYER_NAME).
        setOutputs(NeuralNetConstants.DEFAULT_OUTPUT_LAYER_NAME).
        build());
  }
}
