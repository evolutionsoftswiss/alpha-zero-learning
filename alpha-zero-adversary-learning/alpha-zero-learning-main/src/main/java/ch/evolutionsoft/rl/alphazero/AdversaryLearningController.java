package ch.evolutionsoft.rl.alphazero;

import java.io.IOException;
import java.util.Collections;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import ch.evolutionsoft.rl.AdversaryLearningConfiguration;
import ch.evolutionsoft.rl.AdversaryTrainingExample;
import ch.evolutionsoft.rl.Game;

@SpringBootApplication
@RestController
@RequestMapping("alpha-zero")
public class AdversaryLearningController {

  private static final Logger log = LoggerFactory.getLogger(AdversaryLearningController.class);

  AdversaryLearning adversaryLearning;
  
  @Autowired
  public AdversaryLearningController(AdversaryLearning adversaryLearning) {
  
    this.adversaryLearning = adversaryLearning;
  }
  
  @GetMapping("/isInitialized")
  public Boolean isInitialized() {
    
    log.debug("AdversaryLearning isInitialized {}", adversaryLearning.initialized);
    
    if (adversaryLearning.initialized) {

      log.info("AdversaryLearning initialization completed");
    }
    
    return adversaryLearning.initialized;
  }

  @GetMapping("/trainExamplesList")
  public List<AdversaryTrainingExample> getAdversaryTrainingExamplesList() {

    log.info("Request all training examples from stored history.");    

    List<AdversaryTrainingExample> trainingExamplesList = adversaryLearning.getAllTrainingExamples();
 
    log.info("Adversary learning self plays returned new total {} training examples map",
        trainingExamplesList.size());    

    return trainingExamplesList;
  }

  @GetMapping("/newTrainingExamples")
  public List<AdversaryTrainingExample> getAdversaryTrainingExamples() {

    log.info("Request all training examples from new adversary learning iteration.");    
    try {

      List<AdversaryTrainingExample> allTrainingExamples = adversaryLearning.performIteration();
 
      log.info("Adversary learning self plays returned new total {} training examples",
          allTrainingExamples.size());    

      return allTrainingExamples;

    } catch (IOException ioe) {

      log.error("Error getting actual training examples", ioe);
    }

    log.warn("Continuing with empty list trining examples"); 
    
    return Collections.emptyList();
  }

  @PutMapping("/modelUpdated")
  public Class<Void> modelUpdated() throws IOException {
    
    boolean updatedNetIsUsedForNextIteration = this.adversaryLearning.updateNeuralNet();

    log.info("Updated neural net is used for next iteration {}", updatedNetIsUsedForNextIteration);
    
    return Void.TYPE;
  }

  @GetMapping("/adversaryLearningConfiguration")
  public AdversaryLearningConfiguration getAdversaryLearningConfiguration() {

    log.info("Request AdversaryLearningConfiguration"); 

    return this.adversaryLearning.adversaryLearningConfiguration;
  }

  @GetMapping("/initialGame")
  public Game getInitialGame() {

    log.info("Request initial Game"); 
    
    return this.adversaryLearning.initialGame;
  }
  
}
