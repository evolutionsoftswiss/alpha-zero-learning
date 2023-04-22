package ch.evolutionsoft.rl;

import java.util.Random;

public final class AdversaryLearningConstants {

  public static final long DEFAULT_SEED = 235711;

  /**
   * Non - secure random should be OK here
   */
  @SuppressWarnings("java:S2245")
  public static final Random randomGenerator = new Random(DEFAULT_SEED);

  public static final String DEFAULT_INPUT_LAYER_NAME = "InputLayer";
  public static final String DEFAULT_OUTPUT_LAYER_NAME = "OutputLayer";

  public static final double ZERO = 0.0;
  public static final double ONE = 1.0;
  
  private AdversaryLearningConstants() {
    // Hide constructor
  }
}
