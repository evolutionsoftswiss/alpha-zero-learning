package ch.evolutionsoft.rl4j.tictactoe;

import java.io.IOException;
import java.io.OutputStream;
import java.util.Collection;

import org.deeplearning4j.nn.api.NeuralNetwork;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.rl4j.network.NeuralNet;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;

import ch.evolutionsoft.net.game.tictactoe.TicTacToeConstants;

public class ConvolutionalNeuralNetDQN implements IDQN<ConvolutionalNeuralNetDQN>,
  NeuralNet<ConvolutionalNeuralNetDQN> {

  protected ComputationGraph convolutionalModel;
  
  public ConvolutionalNeuralNetDQN(ComputationGraph model) {
    this.convolutionalModel = model;
  }

  @Override
  public NeuralNetwork[] getNeuralNetworks() {
    
    return new NeuralNetwork[] {convolutionalModel};
  }

  public void save(OutputStream stream) throws IOException {

    ModelSerializer.writeModel(convolutionalModel, stream, true);
  }

  public void save(String path) throws IOException {

    ModelSerializer.writeModel(convolutionalModel, path, true);
  }

  @Override
  public boolean isRecurrent() {

    return false;
  }

  @Override
  public void reset() {
    // no recurrent layer present
  }

  @Override
  public void fit(INDArray input, INDArray labels) {

    fit(input, new INDArray[] {labels});
  }

  @Override
  public void fit(INDArray input, INDArray[] labels) {
    
    convolutionalModel.fit(new INDArray[] {input}, labels);
  }

  @Override
  public INDArray output(INDArray batch) {

    return convolutionalModel.outputSingle(batch);
  }

  @Override
  public INDArray output(Observation observation) {

    return this.output(observation.getData().reshape(
        new int[] {1, TicTacToeConstants.IMAGE_CHANNELS, TicTacToeConstants.IMAGE_SIZE, TicTacToeConstants.IMAGE_SIZE}));
  }

  @Override
  public INDArray[] outputAll(INDArray batch) {

    return convolutionalModel.output(batch);
  }

  @Override
  public ConvolutionalNeuralNetDQN clone() {

    ConvolutionalNeuralNetDQN convolutionalNeuralNetDqnClone =
        new ConvolutionalNeuralNetDQN(convolutionalModel.clone());
    convolutionalNeuralNetDqnClone.convolutionalModel.setListeners(convolutionalModel.getListeners());
    
    return convolutionalNeuralNetDqnClone;
  }

  @Override
  public void copy(ConvolutionalNeuralNetDQN from) {

    convolutionalModel.setParams(from.convolutionalModel.params());
  }

  @Override
  public Gradient[] gradient(INDArray input, INDArray label) {

    convolutionalModel.setInputs(input);
    convolutionalModel.setLabels(label);
    convolutionalModel.computeGradientAndScore();
    Collection<TrainingListener> iterationListeners = convolutionalModel.getListeners();
    if (iterationListeners != null && !iterationListeners.isEmpty()) {
        for (TrainingListener l : iterationListeners) {
            l.onGradientCalculation(convolutionalModel);
        }
    }

    return new Gradient[] {convolutionalModel.gradient()};
  }

  @Override
  public Gradient[] gradient(INDArray input, INDArray[] labels) {
    
    return gradient(input, labels[0]);
  }

  @Override
  public void applyGradient(Gradient[] gradients, int batchSize) {
    ComputationGraphConfiguration convolutionalConfiguration =
        convolutionalModel.getConfiguration();
    
    int iterationCount = convolutionalConfiguration.getIterationCount();
    int epochCount = convolutionalConfiguration.getEpochCount();
    convolutionalModel.getUpdater().
      update(
          gradients[0],
          iterationCount,
          epochCount,
          batchSize,
          LayerWorkspaceMgr.noWorkspaces());
    convolutionalModel.params().subi(gradients[0].gradient());
    
    Collection<TrainingListener> iterationListeners = convolutionalModel.getListeners();
    if (iterationListeners != null && !iterationListeners.isEmpty()) {
        for (TrainingListener listener : iterationListeners) {
            listener.iterationDone(convolutionalModel, iterationCount, epochCount);
        }
    }
    
    convolutionalConfiguration.setIterationCount(iterationCount + 1);
  }

  @Override
  public double getLatestScore() {

    return convolutionalModel.score();
  }

}
