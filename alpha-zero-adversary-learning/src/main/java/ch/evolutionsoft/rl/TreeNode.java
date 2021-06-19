package ch.evolutionsoft.rl;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.nd4j.linalg.api.ndarray.INDArray;

import ch.evolutionsoft.net.game.NeuralNetConstants;

public class TreeNode {
	
	int move;

	int depth;
	
	int lastColorMove;
	
	int timesVisited = 0;
	
	double qValue = 0; 
	
	double uValue = 0;
	
	double moveProbability;
	
	TreeNode parent;
	
	Map<Integer, TreeNode> children = new HashMap<>();
	
	
	public TreeNode(
	    int move,
	    int colorToMove,
	    int depth,
	    double moveProbability,
	    TreeNode parent){
		
		this.parent = parent;
		this.depth = depth;
		this.move = move;
		this.moveProbability = moveProbability;
		this.lastColorMove = colorToMove;
	}


	void expand(Game game, INDArray previousActionProbabilities, INDArray currentBoard) {
	  
	  for (int index : game.getEmptyFields(currentBoard)) {
	    
	    if (!this.children.containsKey(index)) {
	      
	      this.children.put(
	          index,
	          new TreeNode(
	              index,
	              game.getOtherPlayer(this.lastColorMove),
	              this.depth + 1,
	              previousActionProbabilities.getDouble(index),
	              this));
	    }
	  }
	}
	
	protected TreeNode selectMove(double cpUct) {

	  // Handle never visited children
	  List<TreeNode> neverVisitedChildren = new ArrayList<>();
      for (TreeNode treeNode : this.children.values()) {
        
        if (0 >= treeNode.timesVisited) {
          neverVisitedChildren.add(treeNode);
        }
      }
      
      if (!neverVisitedChildren.isEmpty()) {
        
        return neverVisitedChildren.get(
            NeuralNetConstants.randomGenerator.nextInt(
                neverVisitedChildren.size())
            );
      }
	  
      double bestValue = Integer.MIN_VALUE;
      TreeNode bestNode = null;
      
      for (TreeNode treeNode : this.children.values()) {
        	
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
	
	
	public void update(double newValue){
	  
	  this.timesVisited++;
	  
	  this.qValue += (newValue - this.qValue) / this.timesVisited; 
	}
	
	
	public void updateRecursiv(double newValue) {

	  if (null != this.parent) {
	    
	    this.parent.updateRecursiv(1 - newValue);
	  }
	  
	  this.update(newValue);
	}
}
