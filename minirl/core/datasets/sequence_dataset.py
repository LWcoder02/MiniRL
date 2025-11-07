from minirl.core.dataset import Dataset, DatasetInfo


class SequenceDataset(Dataset):
    def __init__(self, environment_info, num_steps = None, num_episodes = None) -> None:
        self._env_info = environment_info
        dataset_info = DatasetInfo.create_dataset_info(environment_info=self._env_info)

        if num_steps is not None:
            num_samples = num_steps
        else:
            horizon = self._dataset_info.horizon
            num_samples = horizon * num_episodes

        base_shape = (num_samples,)
        dtype = None
        field_specs = {
            "state": ((num_samples,) + dataset_info.state_shape, dtype),
            "action": ((num_samples,) + dataset_info.action_shape, int),
            "reward": ((num_samples,), dtype),
            "next_state": ((num_samples,) + dataset_info.state_shape, dtype),
            "terminated": ((num_samples,), dtype)
        }

        super().__init__(dataset_info=dataset_info,
                         field_specs=field_specs,
                         base_shape=base_shape)


    def append(self, state, action, reward, next_state, done, info) -> None:
        super().append(state=state,
                     action=action,
                     reward=reward,
                     next_state=next_state,
                     terminated=done)
        # update info manually

    @classmethod
    def generate(cls, environment_info, num_steps = None, num_episodes = None):
        return cls(environment_info, num_steps, num_episodes)