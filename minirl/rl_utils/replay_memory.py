from minirl.core.dataset import Dataset, DatasetInfo
from minirl.core.agent import AgentInfo
from minirl.core.environment import EnvironmentInfo


class ReplayBuffer():
    def __init__(self, environment_info: EnvironmentInfo, agent_info: AgentInfo,
                 max_size: int = 5000, initial_size: int = 50):
        self._initial_size = initial_size
        self._max_size: int = max_size
        self._idx: int = 0
        self._environment_info = environment_info
        self._agent_info = agent_info

        self.reset()


    def sample(self, batch_size: int):
        indices = self._dataset._backend.randint(0, len(self._dataset), (batch_size,))

        batch = self._dataset[indices]
        return batch


    def add(self, dataset: Dataset):

        state, action, reward, next_state, dones = dataset.parse(to=self._agent_info.backend)
        i = 0
        while i < len(dataset):
            if self._full:
                self._dataset.state[self._idx] = state[i]
                self._dataset.action[self._idx] = action[i]
                self._dataset.reward[self._idx] = reward[i]
                self._dataset.next_state[self._idx] = next_state[i]
                self._dataset.done[self._idx] = dones[i]
            else:
                sample = [state[i], action[i], reward[i], next_state[i], dones[i]]
                self._dataset.append(sample)

            self._idx += 1
            if self._idx == self._max_size:
                self._full = True
                self._idx = 0
            
            i += 1


    def get(self, num_samples: int):
        idxs = self._dataset._backend.randint(0, len(self._dataset), (num_samples,))
        samples = self._dataset[idxs]
        return samples


    def reset(self):
        self._idx = 0
        self._full = False
        replay_memory_info: DatasetInfo = DatasetInfo.create_dataset_info(self._environment_info)
        self._dataset: Dataset = Dataset(replay_memory_info, num_steps=self._max_size)

        
    @property
    def init(self):
        return self.size > self._initial_size
    

    @property
    def size(self):
        return self._idx if not self._full else self._max_size