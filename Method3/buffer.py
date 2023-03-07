import numpy as np

class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size, agent_id):
        max_mem = min(self.mem_cntr, self.mem_size)

        Batch = []
        while len(Batch) < batch_size:
            extra_batch = np.random.choice(max_mem, batch_size * 10, replace=False)
            for idx, state in enumerate(self.state_memory[extra_batch]):
                if state[(agent_id * 6) + 1] >= 0.0:
                    Batch.append(extra_batch[idx])
                    if len(Batch) >= batch_size:
                        break

        states = self.state_memory[Batch]
        states_ = self.new_state_memory[Batch]
        actions = self.action_memory[Batch]
        rewards = self.reward_memory[Batch]
        dones = self.terminal_memory[Batch]

        return states, actions, rewards, states_, dones
