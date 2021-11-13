import gym
import numpy as np
import matplotlib.pyplot as plt


env = gym.make("CartPole-v1")

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 10000
SHOW_EVERY = 500
PLOT_EVERY = 100
# convert continuous observation space and states to discrete size

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

epsilon = 1
START_EPSILON_DECAYING = 1
END_EPSILON_DECAY = EPISODES // 2
epsilon_decay_value = epsilon/(END_EPSILON_DECAY - START_EPSILON_DECAYING)


def create_bins_and_q_table():
    num_bins = 20
    obs_space_size = len(env.observation_space.high)
    bins = [
        np.linspace(-4.8, 4.8, num_bins),
        np.linspace(-4, 4, num_bins),
        np.linspace(-.418, .418, num_bins),
        np.linspace(-4, 4, num_bins)
    ]

    q_table = np.random.uniform(low=-2, high=0, size=([num_bins] * obs_space_size + [env.action_space.n]))

    return bins, obs_space_size, q_table


# metrics
ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}


def get_discrete_state(state, bins, obs_space_size ):
    state_index = []
    for i in range(obs_space_size):
        state_index.append(np.digitize(state[i], bins[i]) - 1) # -1 will turn bin into index
    return tuple(state_index)


bins, obs_space_size, q_table = create_bins_and_q_table()

for episode in range(EPISODES):
    episode_reward = 0
    discrete_state = get_discrete_state(env.reset(), bins, obs_space_size )
    done = False
    cnt = 0

    if episode % SHOW_EVERY == 0:
        print(episode)
        render = True
    else:
        render = False

    while not done:
        cnt += 1
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, done, _ = env.step(action)
        episode_reward += reward
        new_discrete_state = get_discrete_state(new_state, bins, obs_space_size)
        max_future_q = np.max(q_table[new_discrete_state])
        current_q = q_table[discrete_state + (action, )]

        if render:
            env.render()
        if done and cnt < 200:
            reward = -375

        new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        q_table[discrete_state + (action, )] = new_q

        discrete_state = new_discrete_state

    if END_EPSILON_DECAY >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

    ep_rewards.append(episode_reward)

    if not episode % SHOW_EVERY:
        average_reward = sum(ep_rewards[-SHOW_EVERY:])/len(ep_rewards[-SHOW_EVERY: ])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))

        print(f"Episode: {episode} avg: {average_reward} min: {min(ep_rewards[-SHOW_EVERY:])} max: {max(ep_rewards[-SHOW_EVERY:])} ")

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="avg")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max")
plt.legend(loc=4)
plt.show()

env.close()


