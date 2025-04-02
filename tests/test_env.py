from minirl.environments.grid_worlds import GridWorld

def test_environemt():
    grid_world = GridWorld()
    obs, info = grid_world.reset()
    print(obs)