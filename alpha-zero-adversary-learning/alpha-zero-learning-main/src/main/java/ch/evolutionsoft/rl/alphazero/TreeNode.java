package ch.evolutionsoft.rl.alphazero;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ch.evolutionsoft.rl.Game;

public class TreeNode implements Serializable {
  
  transient Logger logger = LoggerFactory.getLogger(TreeNode.class);
	
	int lastMove;

	int depth;
	
	int currentMoveColor;
	
	int timesVisited = 0;
	
	double qValue = Game.DRAW; 
	
	double uValue = 0;
	
	double moveProbability = 0;
	
	TreeNode parent;
	
	transient ConcurrentMap<Integer, TreeNode> children = new ConcurrentHashMap<>();
	
	
	public TreeNode(
	    int lastMove,
	    int currentMoveColor,
	    int depth,
	    double moveProbability,
	    double initialQ,
	    TreeNode parent){

	  this.parent = parent;
	  this.qValue = initialQ;
	  this.depth = depth;
	  this.lastMove = lastMove;
	  this.moveProbability = moveProbability;
	  this.currentMoveColor = currentMoveColor;

	}
	
	void expand(Game game, INDArray previousActionProbabilities) {

  	  Set<Integer> validMoveIndices = game.getValidMoveIndices();
   
  	  for (int moveIndex : validMoveIndices) {
  	      
  	    this.children.put(
  	        moveIndex,
  	        new TreeNode(
  	            moveIndex,
  	            game.getOtherPlayer(this.currentMoveColor),
  	            this.depth + 1,
  	            previousActionProbabilities.getDouble(moveIndex),
  	            Game.getInversedResult(this.qValue),
  	            this));
  	    }
	}
	
	protected TreeNode selectMove(double cpUct) {

	  List<TreeNode> childNodes = new ArrayList<>(this.children.values());

    double bestValue = Integer.MIN_VALUE;
    TreeNode bestNode = null;
    
    for (TreeNode treeNode : childNodes) {
      	
      double currentValue = treeNode.getValue(cpUct);
        
      if (currentValue > bestValue) {
          	
        bestValue = currentValue;
        bestNode = treeNode;
      }
    }

    return bestNode;
	}
	
	
	public double getValue(double cpUct) {

	  this.uValue = 
          cpUct * this.moveProbability *
              Math.sqrt(this.parent.timesVisited) / (1 + this.timesVisited);
	  
	  return this.qValue + this.uValue;
	}

	
	public void update(double newValue) {

	  this.timesVisited++;

	  this.qValue = this.qValue + ((newValue - this.qValue) / (this.timesVisited));
	}
  
  
  public void updateRecursiv(double newValue, TreeNode currentRoot) {

    if (this != currentRoot) {
      
      this.parent.updateRecursiv(Game.getInversedResult(newValue), currentRoot);
    }
    
    this.update(newValue);
  }

  
  public void incrementVisited() {

    this.timesVisited++;
  }
  
  
  public void updateRecursivVisited(TreeNode currentRoot) {

    if (this != currentRoot) {
      
      this.parent.updateRecursivVisited(currentRoot);
    }
    
    this.incrementVisited();
  }

  public boolean isExpanded() {
    return !this.children.isEmpty();
  }

	public Collection<TreeNode> getChildren() {
  
	  return this.children.values();
	}
	
	public TreeNode getChildWithMoveIndex(int moveIndex) {

	  return this.children.get(moveIndex);
	}
	
	public boolean containsChildMoveIndex(int moveIndex) {
     
	  return this.children.containsKey(moveIndex);
	}
}
