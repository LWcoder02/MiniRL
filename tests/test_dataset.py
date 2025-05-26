import sys
sys.path.insert(0, '../')

from minirl.core.dataset import Dataset
from minirl.core.datasets.numpy_dataset import NumpyDataset
from minirl.core.agent import AgentInfo
from minirl.core.environment import EnvironmentInfo

from gymnasium import spaces
import numpy as np

#################### UTILITY START ####################
def create_mdp_info():
    state_space = spaces.Box(low=0, high=9, shape=(2,), seed=0)
    action_space = spaces.Discrete(n=4, seed=0)
    return EnvironmentInfo(action_space=action_space, observation_space=state_space)


def create_agent_info():
    agent_info = AgentInfo("numpy")
    return agent_info


def create_samples(num_samples, obs_space: spaces.Box, action_space: spaces.Discrete):
    samples = []
    for i in range(num_samples):
        state = obs_space.sample()
        action = action_space.sample()
        reward = np.random.randint(low=0, high=5)
        next_state = obs_space.sample()
        done = np.random.choice([True, False])
        samples.append([state, action, reward, next_state, done])

    return samples

def generate_dataset(env_info, agent_info):
    dataset = Dataset.generate(env_info, agent_info, num_steps=5)
    return dataset


def pretty_print(lst):
    lst_string = ""
    for i, item in enumerate(lst):
        lst_string += f"({i}):{item}\n"
    print(lst_string)
#################### UTILITY END ####################


#################### TESTS START ####################

def append_test(dataset: Dataset, samples):
    pretty_print(samples)
    for sample in samples:
        dataset.append(sample)
    

#################### TESTS END ####################


# Main test running function
def run_test(test_type):
    """
    Main test running function. \n
    ``test_type`` specifies the test to be run
    """
    env_info = create_mdp_info()
    agent_info = create_agent_info()
    dataset = generate_dataset(env_info, agent_info)

    if test_type == "generate":
        print(dataset)
    elif test_type == "append":
        samples = create_samples(3, env_info.observation_space, env_info.action_space)
        append_test(dataset, samples)
        print(dataset)

    elif test_type == "get":
        samples = create_samples(5, env_info.observation_space, env_info.action_space)
        append_test(dataset, samples)
        print(dataset[0])


if __name__ == '__main__':
    np.random.seed(0)
    test_type = "get"
    run_test(test_type=test_type)
    