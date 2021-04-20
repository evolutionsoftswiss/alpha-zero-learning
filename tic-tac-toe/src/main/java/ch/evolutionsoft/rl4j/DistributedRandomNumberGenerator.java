package ch.evolutionsoft.rl4j;

import java.util.HashMap;
import java.util.Map;

import org.nd4j.linalg.api.ndarray.INDArray;

public class DistributedRandomNumberGenerator {

  private Map<Integer, Float> distribution;
  private double distSum;

  public DistributedRandomNumberGenerator() {

    distribution = new HashMap<>();
  }

  public DistributedRandomNumberGenerator(INDArray distribution) {

    this();
    for (int index = 0; index < distribution.size(0); index++) {
      this.addNumber(index, distribution.getFloat(index));
    }
  }

  public void addNumber(int value, float distribution) {

    if (this.distribution.get(value) != null) {
      distSum -= this.distribution.get(value);
    }
    this.distribution.put(value, distribution);
    distSum += distribution;
  }

  public int getDistributedRandomNumber() {

    double rand = Math.random();
    double ratio = 1.0f / distSum;
    double tempDist = 0;
    for (Integer i : distribution.keySet()) {
      tempDist += distribution.get(i);
      if (rand / ratio <= tempDist) {
        return i;
      }
    }
    return 0;
  }

}