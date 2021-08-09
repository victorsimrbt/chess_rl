from tensorflow import keras
from variable_settings import *
from chess_env import * 
import numpy as np
from q_network import *

env = ChessEnv()
for _ in range(iterations):
    state = np.array(env.reset())
    episode_reward = 0
    len_episodes += 1
    for timestep in range(1, max_steps_per_episode):
        frame_count += 1

        if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
            move,action = model.explore(env)
        else:
            move,action = model.predict(env)
            
        epsilon -= epsilon_interval / epsilon_greedy_frames
        epsilon = max(epsilon, epsilon_min)
        
        env.step(move)

        if frame_count % update_after_actions == 0 and len(env.done_history) > batch_size:
            state_samples,masks,updated_q_values = env.update_q_values()
            
            for i in range(len(state_samples)):
                with tf.GradientTape() as tape:
                    q_values = model.model(state_samples[i])
                    q_action = tf.reduce_sum(tf.multiply(q_values, masks[i]), axis=1)
                    loss = loss_function(updated_q_values[i], q_action)

                grads = tape.gradient(loss, model.model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.model.trainable_variables))

        if frame_count % update_target_network == 0:
            model_target.model.set_weights(model.model.get_weights())
            white_reward,black_reward = (np.mean(env.episode_reward_history['white']),
                                         np.mean(env.episode_reward_history['white']))
            template = "episode {}, frame count {}, white_reward {}, black_reward {}"
            print(template.format(episode_count, frame_count,white_reward,black_reward))
            
        env.episode_reward_history.append(episode_reward)
        if env.done:
            break

    episode_count += 1