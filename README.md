# Reinforcement-Learning

## Задача стабилизации подвижного маятника на тележке.

Рассматривается задача стабилизации перевернутого маятника, расположенного на подвижной платформе. Маятник удерживается в перевернутом состоянии за счет изменения скорости тележки.

![Image alt](https://github.com/AntonLedyaev/Reinforcement-Learning/raw/main/img/cartpole.gif)


## Описание среды.

Состояние тележки описывается следующими параметрами:

* позиция тележки – значение в диапазоне [-4.8, 4.8];
* скорость тележки;
* угол отклонения шеста от вертикали – значение в диапазоне [-24°, 24°];
* скорость изменения угла наклона шеста.

Действие принимает два значения - 0 и 1:

* 0 – толкнуть тележку влево (приложить к тележке горизонтальную силу, равную +1);
* 1 – толкнуть тележку вправо (приложить к тележке горизонтальную силу, равную -1).

Эпизод завершается если:

* угол шеста вышел из диапазона [-24°, 24°];
* позиция тележки вышла из допустимого диапазона [-4.8, 4.8];
* длина эпизода превышает 500;


## Q-learning
Q-learning это не связанный с политикой без модельный алгоритм ОП, основанный на хорошо известном уравнении Беллмана:

![Image alt](https://github.com/AntonLedyaev/Reinforcement-Learning/raw/main/img/bellman1.png)

Мы можем переписать это уравнение в форме Q-value:

![Image alt](https://github.com/AntonLedyaev/Reinforcement-Learning/raw/main/img/bellman2.png)

Оптимальное значение Q, обозначенное как Q*, может быть выражено как:

![Image alt](https://github.com/AntonLedyaev/Reinforcement-Learning/raw/main/img/bellman.png)

Цель состоит в том, чтобы максимизировать Q-значение.

### Гиперпараметры

* LEARNING_RATE = 0.1
* DISCOUNT = 0.95
* EPISODES = 10000
* epsilon = 1
* START_EPSILON_DECAYING = 1
* END_EPSILON_DECAY = EPISODES // 2

### Результаты:

* min - минимальный score
* avg - средний
* max - максимальный

![Image alt](https://github.com/AntonLedyaev/Reinforcement-Learning/raw/main/img/plotq.jpg)


## DQN

Алгоритм DQN использует нейронную сеть для оценки значений Q-функции Беллмана. На вход сети подаются текущие кадры игрового поля, а выходом - соответствующее значение Q для каждого возможного действия. Нейросеть обучается так же, как и в случае q-learning, обновляя значения Q-функции Беллмана. 

![Image alt](https://github.com/AntonLedyaev/Reinforcement-Learning/raw/main/img/dqn_bellma.png)

Здесь:
* φ эквивалентно состоянию s
* θ обозначает параметры в нейронной сети

### Гиперпараметры:
* EPISODES = 1000
* GAMMA = 0.95
* EPSILON = 1.0
* EPSILON_MIN = 0.001
* EPSILON_DECAY = 0.999
* batch_size = 64
* train_start = 1000

### Результаты: 


![Image alt](https://github.com/AntonLedyaev/Reinforcement-Learning/raw/main/img/plot.jpg)

![Image alt](https://github.com/AntonLedyaev/Reinforcement-Learning/raw/main/img/cartpole_example.gif)



