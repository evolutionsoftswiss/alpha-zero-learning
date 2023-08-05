Java Alpha Zero Reinforcement Learning with deeplearning4j
==========================================================
Alpha zero learning is a Java implementation of the alpha zero algorithm using deeplearning4j library.

## Introduction
There are already several alpha[Go] zero related projects on github in python and also c++.
The Java implementation here could help to reuse your existing Java game logics with alpha zero reinforcement learning.
It may also be a simpler approach to alpha zero for people more familiar with Java than python. 

### Alpha[Go] Zero algorithm
During almost 20 years after IBM DeepBlue defeated Kasparov 1997 in chess, 19x19 Go was still a game where computers were far from human strength level of play. It changed definitely after AlphaGo vs. Lee Sedol, the Google Deepmind Challenge Match. Alpha Go has beaten the worlds best Go Player in 5 Games 4:1. There is a very interesting Movie on Youtube around the event: [AlphaGo - The Movie](https://www.youtube.com/watch?v=WXuK6gekU1Y). Beside the DeepMind efforts it also gives an insight about the ideology and philosophy of the Go Game in asian countries.

The AlphaGo algorithm was further adapted and improved with AlphaGo Zero [Mastering the game of Go without human knowledge](https://www.nature.com/articles/nature24270) and Alpha Zero [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/pdf/1712.01815.pdf).

### Why there are not thousands of strong artificial Go intelligences now ?
It is not enough to know the algorithm and having an implementation ready. A rough calculation [Computer-Go zero performance](http://web.archive.org/web/20190205013627/http://computer-go.org/pipermail/computer-go/2017-October/010307.html) estimates the training time to generate as much positions as AlphaGo zero could take 1700 years on a single machine with standard hardware.

There are efforts like [leela-zero](https://github.com/leela-zero/leela-zero) to repeat the training efforts in a public and distributed manner. Similarly [minigo](https://github.com/tensorflow/minigo) tried to reproduce the learning progress.

You can more easily adapt the algorithm to less complex board games like connect four, Gomoku, Othello and others. With such games it is more realistic to perform the necessary training to obtain a strong artificial intelligence.

## Using Java Alpha Zero
The goal of Java alpha-zero-learning is to enable alpha zero for less complex games. The implementation does not support distributed learning. It is designed to run on a single machine with optional graphic cards usage.

### Generic release build
You can use the existing Java alpha-zero-learning with the generic published release builds. With ch.evolutionsoft.rl.alphazero.tictactoe-2.0.0-jar-with-dependencies you can directly repeat the training for the Tic Tac Toe prototype. See also the submodule [tic-tac-toe/README.md](./tic-tac-toe/README.md) for a few more information.

ch.evolutionsoft.rl.alphazero.adversary-learning-2.0.0-jar-with-dependencies would let you reuse the general part of the implementation for other board games. The submodule [alpha-zero-adversary-learning/README.md](./alpha-zero-adversary-learning/README.md) contains hints about a new board game implementation.

### Running the Connect Four implementation


### Running the Tic Tac Toe implementation
You should run the different main methods from within the same directory. The alpha zero learning files, bestmodel.bin and trainexamples.obj, are searched or created in the current directory.

If you download the generic release build without doing a local mvn build, your -cp classpath value needs to point to the correct location of the jar file with dependencies.

Change to the tic-tac-toe submodule directory and execute one of the following commands.

#### Evaluation method

	~/git/alpha-zero-learning/tic-tac-toe$ java -cp target/ch.evolutionsoft.rl.alphazero.tictactoe-1.1.1-jar-with-dependencies.jar ch.evolutionsoft.rl.tictactoe.TicTacToeGamesMain

#### Restart the learning progress from scratch

	~/git/alpha-zero-learning/tic-tac-toe$ java -cp target/ch.evolutionsoft.rl.alphazero.tictactoe-1.1.1-jar-with-dependencies.jar ch.evolutionsoft.rl.tictactoe.TicTacToeReinforcementLearningMain
	
#### Continue the learning progress from an iteration
By changing the iterationStart to a value greater than 1, you can continue a training progress. The latest bestmodel.bin and trainExamples.obj are loaded from the current directory

	    AdversaryLearningConfiguration adversaryLearningConfiguration =
        new AdversaryLearningConfiguration.Builder().
        iterationStart(251).
        build();

And then after rebuilding with 'mvn package':        

		~/git/alpha-zero-learning/tic-tac-toe$ java -cp target/ch.evolutionsoft.rl.alphazero.tictactoe-1.1.1-jar-with-dependencies.jar ch.evolutionsoft.rl.tictactoe.TicTacToeReinforcementLearningMain
	

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
Refer to the submodule [alpha-zero-adversary-learning/README.md](alpha-zero-adversary-learning/README.md) to see what's necessary for a new game implementation.

## Implementation details

### Simplified parallelization
#### Multi threaded Monte Carlo Tree Search with separate trees
The Monte Carlo Tree Search implementation [MonteCarloTreeSearch.java](https://github.com/evolutionsoftswiss/alpha-zero-learning/blob/master/alpha-zero-adversary-learning/alpha-zero-learning-main/src/main/java/ch/evolutionsoft/rl/alphazero/MonteCarloTreeSearch.java) is run separately for each thread. The separate trees help to produce more different samples, but are probably less accurate than a single shared tree.

### Mvn build and packaging
The mvn builds for each submodule take several minutes and a lot of different system architecture dependencies are packaged into the jar's with dependencies. That is deeplearning4j related and leading to very distribution packages.
