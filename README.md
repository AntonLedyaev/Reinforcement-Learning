# Reinforcement-Learning

## Cart-Pole Stabilization Problem.

This project considers the problem of stabilizing an inverted pendulum mounted on a moving cart. The pendulum is kept in the upright position by controlling the velocity of the cart.

![Image alt](https://github.com/AntonLedyaev/Reinforcement-Learning/raw/main/img/cartpole.gif)


## Environment Description

The state of the cart is described by the following parameters:

* cart position – value in the range [-4.8, 4.8];
* cart velocity;
* pole angle from the vertical – value in the range [-24°, 24°];
* angular velocity of the pole.

The action takes two possible values: 0 and 1:

* 0 – push the cart to the left (apply a horizontal force of +1);
* 1 – 1 – push the cart to the right (apply a horizontal force of -1).

The episode terminates if:

* the pole angle goes out of the range [-24°, 24°];
* the cart position goes out of the allowed range [-4.8, 4.8];
* the episode length exceeds 500 steps.


## Q-learning
Q-learning is an off-policy, model-free reinforcement learning algorithm based on the well-known Bellman equation:

![Image alt](https://github.com/AntonLedyaev/Reinforcement-Learning/raw/main/img/bellman1.png)

We can rewrite this equation in terms of the Q-value:

![Image alt](https://github.com/AntonLedyaev/Reinforcement-Learning/raw/main/img/bellman2.png)

The optimal Q-value, denoted as Q*, can be expressed as:

![Image alt](https://github.com/AntonLedyaev/Reinforcement-Learning/raw/main/img/bellman.png)

The goal is to maximize the Q-value.

### Hyperparameters

* LEARNING_RATE = 0.1
* DISCOUNT = 0.95
* EPISODES = 10000
* epsilon = 1
* START_EPSILON_DECAYING = 1
* END_EPSILON_DECAY = EPISODES // 2

### Results:

* min - minimum score
* avg - average score
* max - maximum score

![Image alt](https://github.com/AntonLedyaev/Reinforcement-Learning/raw/main/img/plotq.png)


## DQN

The DQN algorithm uses a neural network to approximate the Q-function values from the Bellman equation. The input to the network is the current state (frames of the environment), and the output is the Q-value for each possible action. The neural network is trained similarly to Q-learning by updating the Bellman Q-function values.

![Image alt](https://github.com/AntonLedyaev/Reinforcement-Learning/raw/main/img/dqn_bellma.png)

Здесь:
* φ corresponds to the state s
* θ represents the parameters of the neural network

### Hyperparameters:
* EPISODES = 1000
* GAMMA = 0.95
* EPSILON = 1.0
* EPSILON_MIN = 0.001
* EPSILON_DECAY = 0.999
* batch_size = 64
* train_start = 1000

### Results: 

* min - minimum score
* avg - average score
* max - maximum score

![Image alt](https://github.com/AntonLedyaev/Reinforcement-Learning/raw/main/img/plot.jpg)

![Image alt](https://github.com/AntonLedyaev/Reinforcement-Learning/raw/main/img/cartpole_example.gif)



