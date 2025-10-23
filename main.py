from minirl.approximators.table import Table
import numpy as np

from minirl.algorithms.temporal_difference.q_learning import QLearning
from minirl.environments.grid_worlds import GridWorld
from minirl.policy.td_policy import EpsilonGreedyPolicy
from minirl.core.dataset import Dataset
from minirl.core.agent import Agent
from minirl.utils.minibatch_handling import generate_minibatch




def train_test_agent():
    np.random.seed(0)
    environment = GridWorld()
    policy = EpsilonGreedyPolicy(1.0, decay=0.95, eps_min=0.005)
    agent = QLearning(environment=environment, policy=policy, learning_rate=0.1)
    # agent.learn(num_steps=10_000, num_steps_per_fit=1)
    return agent


def inspect():
    environment = GridWorld()
    policy = EpsilonGreedyPolicy(1.0)
    agent = QLearning(environment=environment, policy=policy, learning_rate=0.001)
    dataset = Dataset(environment.get_environment_info(), agent_info=agent._agent_info, num_steps=10)
    agent._reset(initial_state=None)
    for i in range(10):
        sample = agent._step()
        dataset.append(sample=sample[:-1]) 
    # print(dataset)
    for batch in generate_minibatch(4, dataset):
        print(batch)


def train_and_evaluate():
    environment = GridWorld()
    agent = train_test_agent()
    print(agent.approximator)

    done = False
    state, info = environment.reset()
    environment.render()
    i = 0
    while not done:
        action = agent.draw_action(state)
        sample = environment.step(action)
        state = sample[0]
        environment.render()
        done = sample[-2]
        i += 1
    print(i)





def main():
    inspect()


if __name__ == '__main__':
    main()