from tensorflow import keras
gamma = 0.99  
epsilon = 1  
epsilon_min = 0.1  
epsilon_max = 1.0  
epsilon_interval = (
    epsilon_max - epsilon_min
)  
batch_size = 32  
max_steps_per_episode = 200
num_actions = 4096
optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

running_reward = 0
episode_count = 0
frame_count = 0

epsilon_random_frames = 50000
epsilon_greedy_frames = 1000000.0
max_memory_length = 100000
update_after_actions = 4
update_target_network = 100
loss_function = keras.losses.Huber()
len_episodes = 0
iterations = 1000000