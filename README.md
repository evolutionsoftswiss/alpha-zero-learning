# Java Alpha Zero Reinforcement Learning
Alpha zero learning is a Java implementation of the alpha zero algorithm using deeplearning4j library.

## Introduction
There are already several alpha[Go] zero related projects on github in python and also c++.
The Java implementation here could help to reuse your existing Java game logics with alpha zero reinforcement learning.
It may also be a simpler approach to alpha zero for people more familiar with Java than python, like me. 

### Alpha[Go] Zero algorithm
During almost 20 years after IBM DeepBlue defeated Kasparov 1997 in chess, 19x19 Go was still a game where computers were far from human strength level of play. It changed definitely after AlphaGo vs. Lee Sedol, the Google Deepmind Challenge Match. Alpha Go has beaten the worlds best Go Player in 5 Games 4:1. There is a very interesting Movie on Youtube around the event: [AlphaGo - The Movie](https://www.youtube.com/watch?v=WXuK6gekU1Y). Beside the DeepMind efforts it also gives an insight about the ideology and philosophy of the Go Game in asian countries.

The AlphaGo algorithm was further adapted and improved with AlphaGo Zero [Mastering the game of Go without human knowledge](https://www.nature.com/articles/nature24270) and Alpha Zero [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/pdf/1712.01815.pdf).

### Why there are not tons of strong Go AI's now ?
It is not enough to know the algorithm and having an implementation ready. A rough calculation [Computer-Go zero performance](http://web.archive.org/web/20190205013627/http://computer-go.org/pipermail/computer-go/2017-October/010307.html) estimates the training time to generate as much positions as AlphaGo zero could take 1700 years.

There are efforts like [leela-zero](https://github.com/leela-zero/leela-zero) to repeat the training efforts in a public and distributed manner.

You can more easily adapt the algorithm to less complex board games like connect four, Gomoku, Othello and others. With such games it is more realistic to perform the necessary training to obtain a strong artificial intelligence.

## Using Java Alpha Zero
Also the first goal of Java alpha-zero-learning is to enable alpha zero for less complex games. The implementation does not yet enable distributed learning and even omits some parallelization possibilities on a single machine yet.

### Generic release build
You can use the existing Java alpha-zero-learning with the generic published release builds. With ch.evolutionsoft.rl.alphazero.tictactoe-1.1.0-jar-with-dependencies you can directly repeat the training for the tic-tac-toe prototype. See also the submodule (tic-tac-toe/README.md)[tic-tac-toe/README.md] for a few more information.

ch.evolutionsoft.rl.alphazero.adversary-learning-1.1.0-jar-with-dependencies would let you reuse the general part of the implementation for other board games. The submodule (alpha-zero-adversary-learning/README.md)[alpha-zero-adversary-learning/README.md] contains hints about a new board game implementation.

### Rebuild for your hardware
With deeplearning4j you can use CUDA to perform the neural net model computations on a GPU. You would configure it by replacing the following two dependencies with CUDA dependencies matching your available version.

	<dependency>
		<groupId>org.nd4j</groupId>
		<artifactId>nd4j-native-platform</artifactId>
	</dependency>
	<dependency>
		<groupId>org.nd4j</groupId>
		<artifactId>nd4j-native</artifactId>
	</dependency>

Also without GPU there is the AVX/AVX2 performance improvement on newer CPU's. Use the operating system dependencies for avx to enable it. The logs will show a warning when you're running the generic build on a avx/avx2 supported CPU:


	<dependency>
		<groupId>org.nd4j</groupId>
		<artifactId>nd4j-native</artifactId>
		<version>1.0.0-beta7</version>
		<classifier>windows-x86_64-avx2</classifier>
	</dependency>

### Implement new games
Refer to the submodule (alpha-zero-adversary-learning/README.md)[alpha-zero-adversary-learning/README.md] to see what's necessary for a new game implementation.

## Open implementation issues

### Missing parallelization
#### Single threaded Monte Carlo Tree Search
The Monte Carlo Tree Search implementation [MonteCarloSearch.java](https://github.com/evolutionsoftswiss/alpha-zero-learning/blob/master/alpha-zero-adversary-learning/src/main/java/ch/evolutionsoft/rl/MonteCarloSearch.java) is not yet implemented to support multi-threads. First trials with parallelized playout method let [MonteCarloTreeSearchTest.java](https://github.com/evolutionsoftswiss/alpha-zero-learning/blob/master/alpha-zero-adversary-learning/src/test/java/ch/evolutionsoft/rl/MonteCarloTreeSearchTest.java) fail. The mismatch in real visit counts versus expected visit counts of direct root child nodes shows the existence of problems in parallelized runs.

Without synchronizing almost all of the code that should run in parallel, those issues and the failing test case remained.

#### Single threaded challenge games in AlphaGo variation
Also when challenging an updated Model against the previous model in a number of games, that is done game after game.
That makes the algorithm also slower for a very simple game like tic-tac-toe.

You may want to prefer to always update the model after having it trained with new play-through examples. That approach was also a change from Alpha Zero compared to AlphaGo Zero.

## Potentially next steps in this repository

### Provide a Connect Four game
I've foreseen to provide a connect four implementation here as the next step.

### Trial with 9x9 Go
Maybe I'll give it a try with 9x9 Go after getting a NVIDIA RTX A6000 or similar.