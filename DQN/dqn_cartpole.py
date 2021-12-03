import os
import random
import gym
import numpy as np
from collections import deque
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.optimizers import Adam, RMSprop

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# наша модель, 4 полносвязных слоя по 512, 256, 64, 2 нейрона, оптимизатор RMSProp, можно потестить Adam, loss - MSE
def DQNModel(input_shape, action_space):
    X_input = Input(input_shape)

    X = Dense(512, input_shape=input_shape, activation="relu", kernel_initializer='he_uniform')(X_input)

    X = Dense(256, activation="relu", kernel_initializer='he_uniform')(X)

    X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)

    X = Dense(action_space, activation="linear", kernel_initializer='he_uniform')(X)

    model = Model(inputs=X_input, outputs=X)
    model.compile(loss="mse", optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])

    model.summary()
    return model


# класс нашего агента
class DQNAgent:
    def __init__(self):
        self.env = gym.make('CartPole-v1')
        self.observation_space_size = self.env.observation_space.shape[0]
        self.action_space_size = self.env.action_space.n
        self.EPISODES = 1000
        self.memory = deque(maxlen=2000)
        self.SHOW_EVERY = 100
        self.SUCCEED_COUNT = 0

        self.GAMMA = 0.95
        self.EPSILON = 1.0
        self.EPSILON_MIN = 0.001
        self.EPSILON_DECAY = 0.999
        self.batch_size = 64
        self.train_start = 1000

        # создаем модель
        self.model = DQNModel(input_shape=(self.observation_space_size,), action_space=self.action_space_size)

    # запоминаем то, что мы уже сделали
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.train_start:
            if self.EPSILON > self.EPSILON_MIN:
                self.EPSILON *= self.EPSILON_DECAY

    # выбираем действие
    def act(self, state):
        if np.random.random() <= self.EPSILON:
            return random.randrange(self.action_space_size)
        else:
            return np.argmax(self.model.predict(state))
        
    # проходим по значениям в памяти, вычисляем предикт для текущего состояния и следующего 
    def replay(self):
        if len(self.memory) < self.train_start:
            return
        # заполняем батч рандомыными значениями из памяти
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))

        state = np.zeros((self.batch_size, self.observation_space_size))
        next_state = np.zeros((self.batch_size, self.observation_space_size))
        action, reward, done = [], [], []

        # пробегаем по батчу и заполняем state, action, reward, next_state, done
        for i in range(self.batch_size):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])

        # делаем предикты
        target = self.model.predict(state)
        target_next = self.model.predict(next_state)
        # Вычисление функции беллмана
        for i in range(self.batch_size):

            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.GAMMA * (np.amax(target_next[i]))

        # Обучаем нейросеть по батчам
        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)

    # загрузка модели
    def load(self, name):
        self.model = load_model(name)

    #сохранение модели
    def save(self, name):
        self.model.save(name)

    #обучение модели
    def forward(self):
        for e in range(self.EPISODES):
            if e % self.SHOW_EVERY == 0:
                self.env.render()
            state = self.env.reset()
            state = np.reshape(state, [1, self.observation_space_size])
            done = False
            i = 0
            while not done:
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.observation_space_size])
                if not done or i == self.env._max_episode_steps - 1:
                    reward = reward
                else:
                    reward = -100
                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1
                if done:

                    print("Количество эпизодов: {}/{}, результат: {}, epsilon: {:.2}".format(e, self.EPISODES, i, self.EPSILON))
                    if i == 500:
                        print("Сохраняем обученную модель как: cartpole-dqn.h5")
                        self.save("cartpole-dqn.h5")
                        self.SUCCEED_COUNT += 1
                    if self.SUCCEED_COUNT == 5:
                        return
                self.replay()

    # тестирование обученной модели
    def test(self):
        self.load("cartpole-dqn.h5")
        for e in range(self.EPISODES):
            state = self.env.reset()
            state = np.reshape(state, [1, self.observation_space_size])
            done = False
            i = 0
            while not done:
                self.env.render()
                action = np.argmax(self.model.predict(state))
                next_state, reward, done, _ = self.env.step(action)
                state = np.reshape(next_state, [1, self.observation_space_size])
                i += 1
                if done:
                    print("episode: {}/{}, score: {}".format(e, self.EPISODES, i))
                    break


if __name__ == "__main__":
    agent = DQNAgent()
    #agent.forward()
    agent.test()
