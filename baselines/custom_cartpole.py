# Source: https://github.com/openai/baselines/blob/master/baselines/deepq/experiments/custom_cartpole.py

import gym
import itertools
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import matplotlib.pyplot as plt
import baselines.common.tf_util as U

from baselines import logger
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.common.schedules import LinearSchedule


def model(inpt, num_actions, scope, reuse=False):
    """This model takes as input an observation and returns values of all actions."""
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.relu)
#        out = layers.fully_connected(out, num_outputs=5, activation_fn=tf.nn.tanh)
#        out = layers.fully_connected(out, num_outputs=32, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return out


if __name__ == '__main__':
    learning_rate = 1e-3
    lay = [64]
    n_experiments = 1
    size_expe = 1000
    all_rewards = np.zeros((n_experiments, size_expe))
    with U.make_session(8):
        # Create the environment
        env = gym.make("CartPole-v0")
        # Create all the functions necessary to train the model
        act, train, update_target, debug = deepq.build_train(
            make_obs_ph=lambda name: U.BatchInput(env.observation_space.shape, name=name),
            q_func=model,
            num_actions=env.action_space.n,
            optimizer=tf.train.AdamOptimizer(learning_rate=5e-4),
            #optimizer = tf.train.RMSPropOptimizer(learning_rate)
        )
        # Create the replay buffer
        replay_buffer = ReplayBuffer(50000)
        # Create the schedule for exploration starting from 1 (every action is random) down to
        # 0.02 (98% of actions are selected according to values predicted by the model).
        exploration = LinearSchedule(schedule_timesteps=10000, initial_p=1.0, final_p=0.02)

        for expe in range(n_experiments):
            # Initialize the parameters and copy them to the target network.
            U.initialize()
            update_target()

            episode_rewards = [0.0]
            obs = env.reset()
            for t in itertools.count():
                # Take action and update exploration to the newest value
                action = act(obs[None], update_eps=exploration.value(t))[0]
                new_obs, rew, done, _ = env.step(action)
                # Store transition in the replay buffer.
                replay_buffer.add(obs, action, rew, new_obs, float(done))
                obs = new_obs

                episode_rewards[-1] += rew
                if done:
                    obs = env.reset()
                    episode_rewards.append(0)

                is_solved = t > 100 and np.mean(episode_rewards[-101:-1]) >= 200

                if is_solved:
                    pass
                    # Show off the result
                    #env.render()
                    #continue
                else:
                    # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                    if t > 1000:
                        obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(32)
                        train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
                    # Update target network periodically.
                    if t % 1000 == 0:
                        update_target()

                if done and len(episode_rewards) % 100 == 0:
                    logger.record_tabular("steps", t)
                    logger.record_tabular("episodes", len(episode_rewards))
                    logger.record_tabular("mean episode reward", round(np.mean(episode_rewards[-101:-1]), 1))
                    logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                    logger.dump_tabular()

                if len(episode_rewards) == size_expe + 1:
                    all_rewards[expe] = episode_rewards[:-1]
                    break
    np.savetxt('results.txt', all_rewards)
    mean_r = np.mean(all_rewards, axis=0)
    std = np.std(all_rewards, axis = 0)
    plt.figure(1)
    plt.clf()
    plt.plot(mean_r)
    plt.fill_between(np.arange(size_expe), mean_r - std, mean_r + std, alpha=0.2, color='b')
    plt.xlabel('Episode')
    plt.ylabel('Average reward / episode')
    plt.title('Cartpole performance with DQN (RMSProp, l_r={}, layers = {})'.format(learning_rate, lay))
    plt.savefig('DQN.eps')
    plt.show()
