package ch.evolutionsoft.rl;

import java.io.IOException;
import java.util.Collections;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class AdversaryLearningController {

  private static final Logger log = LoggerFactory.getLogger(AdversaryLearningController.class);

  AdversaryLearning adversaryLearning;
  
  public AdversaryLearningController(AdversaryLearning adversaryLearning) {
  
    this.adversaryLearning = adversaryLearning;
  }

  public List<AdversaryTrainingExample> getAdversaryTrainingExamples() {
    
    try {
      return adversaryLearning.performIteration();
    } catch (IOException ioe) {
      log.error("Error getting actual training examples", ioe);
    }
    
    return Collections.emptyList();
  }

  public void modelUpdated() throws IOException {
    
    this.adversaryLearning.updateNeuralNet();
  }
  
  public AdversaryLearningConfiguration getAdversaryLearningConfiguration() {

    return this.adversaryLearning.adversaryLearningConfiguration;
  }
  
  public Game getInitialGame() {
    
    return this.adversaryLearning.initialGame;
  }
  
}
