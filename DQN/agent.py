import tensorflow as tf
from tensorflow.keras.layers import Dense
import gym
from collections import deque
import random
import numpy as np
import datetime
import statistics

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'logs/' + current_time
summary_writer = tf.summary.create_file_writer(log_dir)


class ReplayBuffer:
    def __init__(self, capacity=100000, batch_size=32) -> None:
        self.buffer = deque(maxlen=capacity)
        self.batch_size = batch_size

    def put(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])

    def sample(self):
        sample = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, done = map(
            np.asarray, zip(*sample))
        states = np.array(states).reshape(self.batch_size, -1)
        next_states = np.array(next_states).reshape(self.batch_size, -1)
        return states, actions, rewards, next_states, done

    def size(self):
        return len(self.buffer)


class ActionValueModel(tf.keras.Model):
    def __init__(self, action_dim, name='ActionValueModel') -> None:
        super().__init__(name=name)

        self.dense1 = Dense(32, activation='relu')
        self.dense2 = Dense(16, activation='relu')
        self.dense3 = Dense(action_dim)

    def call(self, inputs):
        h = self.dense1(inputs)
        h = self.dense2(h)
        return self.dense3(h)


class Agent:
    def __init__(self, env, num_replay=10) -> None:
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        self.num_replay = num_replay

        self.model = ActionValueModel(self.action_dim)
        self.target_model = ActionValueModel(self.action_dim)
        self.target_update()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.buffer = ReplayBuffer()

        self.gamma = 0.95
        self.epsilon = 1.0
        self.eps_decay = 0.995
        self.eps_min = 0.01

    def target_update(self):
        weights = self.model.get_weights()
        self.target_model.set_weights(weights)

    def get_action(self, state):
        state = np.reshape(state, [1, self.state_dim])
        self.epsilon *= self.eps_decay
        self.epsilon = max(self.epsilon, self.eps_min)
        q_value = self.model.predict(state)[0]
        if np.random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        return np.argmax(q_value)

    def optimize(self):
        states, actions, rewards, next_states, done = self.buffer.sample()
        next_q_values = self.target_model(next_states)
        next_q_action = rewards + (1 - done) * tf.reduce_max(
            next_q_values, axis=1) * self.gamma
        with tf.GradientTape() as tape:
            q_values = self.model(states)
            masks = tf.one_hot(actions, self.action_dim)
            q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            loss = self.loss_fn(next_q_action, q_action)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables))

    def train(self, max_episodes=1000):

        # Cartpole-v0 is considered solved if average reward is >= 195 over 100
        # consecutive trials
        reward_threshold = 195
        running_reward = 0
        min_episodes_criterion = 100
        episodes_reward = deque(maxlen=min_episodes_criterion)
        with summary_writer.as_default():
            for ep in range(max_episodes):
                done, total_reward, total_steps = False, 0, 0
                state = self.env.reset()
                while not done:
                    action = self.get_action(state)
                    next_state, reward, done, _ = self.env.step(action)
                    self.buffer.put(state, action, reward, next_state, done)
                    total_reward += reward
                    total_steps += 1
                    state = next_state
                    if self.buffer.size() >= self.buffer.batch_size:
                        for _ in range(self.num_replay):
                            self.optimize()
                        self.target_update()

                episodes_reward.append(total_reward)
                running_reward = statistics.mean(episodes_reward)
                tf.summary.scalar('reward', total_reward, step=ep)
                tf.summary.scalar('running_reward', running_reward, step=ep)
                print(
                    f'EP{ep} reward={total_reward}, running_reward={running_reward}'
                )

                if running_reward > reward_threshold and ep >= min_episodes_criterion:
                    break


def main():
    env = gym.make('CartPole-v1')
    agent = Agent(env)
    agent.train(max_episodes=1000)


if __name__ == "__main__":
    main()