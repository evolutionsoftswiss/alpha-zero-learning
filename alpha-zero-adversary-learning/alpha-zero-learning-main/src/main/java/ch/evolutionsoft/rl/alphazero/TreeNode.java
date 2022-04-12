package ch.evolutionsoft.rl.alphazero;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ch.evolutionsoft.rl.Game;

public class TreeNode {

  final Object lock = new Object();
  
  Logger logger = LoggerFactory.getLogger(TreeNode.class);
	
	volatile int lastMove;

	volatile int depth;
	
	volatile int lastMoveColor;
	
	volatile int timesVisited = 0;
	
	volatile int virtualLosses = 0;
	
	volatile double qValue = AdversaryLearning.DRAW_VALUE; 
	
	volatile double uValue = 0;
	
	volatile double moveProbability;
	
	volatile TreeNode parent;
	
	ConcurrentMap<Integer, TreeNode> children = new ConcurrentHashMap<>();
	
	
	public TreeNode(
	    int lastMove,
	    int lastMoveColor,
	    int depth,
	    double moveProbability,
	    double initialQ,
	    TreeNode parent){

	  this.parent = parent;
	  this.qValue = initialQ;
	  this.depth = depth;
	  this.lastMove = lastMove;
	  this.moveProbability = moveProbability;
	  this.lastMoveColor = lastMoveColor;
	}
	
	void expand(Game game, INDArray previousActionProbabilities) {

    synchronized(lock) {
  	  Set<Integer> validMoveIndices = game.getValidMoveIndices();
   
  	  for (int moveIndex : validMoveIndices) {
  	      
  	    this.children.put(
  	        moveIndex,
  	        new TreeNode(
  	            moveIndex,
  	            game.getOtherPlayer(this.lastMoveColor),
  	            this.depth + 1,
  	            previousActionProbabilities.getDouble(moveIndex),
  	            1 - this.qValue,
  	            this));
  	    }
    }
	}

	public boolean isExpanded() {
	  
	  return !this.children.isEmpty();
	}
	
	protected synchronized TreeNode selectMove(double cpUct) {

      List<TreeNode> childNodes = new ArrayList<>(this.children.values());
      Collections.shuffle(childNodes);

      double bestValue = Integer.MIN_VALUE;
      TreeNode bestNode = null;
      
      for (TreeNode treeNode : childNodes) {
        	
        double currentValue = treeNode.getValue(cpUct);
          
        if (currentValue > bestValue) {
            	
          bestValue = currentValue;
          bestNode = treeNode;
        }
      }

      bestNode.virtualLosses++;
      
      return bestNode;
	}
	
	
	public synchronized double getValue(double cpUct) {
	  
	  this.uValue = 
          cpUct * this.moveProbability *
              Math.sqrt(this.parent.timesVisited) / (1 + this.timesVisited + this.virtualLosses);
	  
	  return this.qValue + this.uValue;
	}
	
	
	public synchronized void update(double newValue) {

	  this.timesVisited++;
	  
	  this.qValue += (newValue - this.qValue) / (this.timesVisited); 

	  this.virtualLosses--;
	}
	
	
	public synchronized void updateRecursiv(double newValue) {
	  

	  if (null != this.parent) {
	    
	    this.parent.updateRecursiv(1 - newValue);
	  }
	  
	  this.update(newValue);
	  
	}

	public TreeNode getChildWithMoveIndex(int moveIndex) {
	  
	  return this.children.get(moveIndex);
	}
	
	public boolean containsChildMoveIndex(int moveIndex) {
	  
	  return this.children.containsKey(moveIndex);
	}
}
