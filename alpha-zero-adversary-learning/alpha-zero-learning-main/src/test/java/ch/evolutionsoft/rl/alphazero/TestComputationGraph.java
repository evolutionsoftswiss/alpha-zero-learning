package ch.evolutionsoft.rl.alphazero;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import ch.evolutionsoft.rl.AdversaryLearningConstants;

public class TestComputationGraph extends ComputationGraph {

  public TestComputationGraph() {

    super(new ComputationGraphConfiguration.GraphBuilder(new NeuralNetConfiguration.Builder()).
        addInputs(AdversaryLearningConstants.DEFAULT_INPUT_LAYER_NAME).setInputTypes(InputType.convolutional(1, 1, 1)).
        addLayer(
            AdversaryLearningConstants.DEFAULT_OUTPUT_LAYER_NAME,
            new OutputLayer.Builder().activation(Activation.SIGMOID).lossFunction(LossFunction.XENT).nOut(1).build(),
            AdversaryLearningConstants.DEFAULT_INPUT_LAYER_NAME).
        setOutputs(AdversaryLearningConstants.DEFAULT_OUTPUT_LAYER_NAME).
        build());
  }
}
