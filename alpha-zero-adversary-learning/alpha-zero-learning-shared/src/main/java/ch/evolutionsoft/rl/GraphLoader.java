package ch.evolutionsoft.rl;

import java.io.IOException;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class GraphLoader {

  public static final Logger log = LoggerFactory.getLogger(GraphLoader.class);

  public static ComputationGraph loadComputationGraph(AdversaryLearningConfiguration adversaryLearningConfiguration) {

    String absoluteBestModelPath =
        AdversaryLearningConfiguration.getAbsolutePathFrom(adversaryLearningConfiguration.getBestModelFileName());

    try {
      ComputationGraph computationGraph = ModelSerializer.restoreComputationGraph(absoluteBestModelPath, true);
      computationGraph.setLearningRate(adversaryLearningConfiguration.getLearningRate());
      if (null != adversaryLearningConfiguration.getLearningRateSchedule()) {
        computationGraph.setLearningRate(adversaryLearningConfiguration.getLearningRateSchedule());
      }
      log.info("restored model {}", absoluteBestModelPath);
    
      return computationGraph;
      
    } catch (IOException ioe) {
      log.error("Error on loading computation graph", ioe);
    }
    
    return null;
  }
  
  private GraphLoader() {
    // empty constructor
  }

}
