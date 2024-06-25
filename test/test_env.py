from minirl.environments.grid_worlds import GridWorld

if __name__ =='__main__':
    grid_world = GridWorld()
    obs, info = grid_world.reset()
    print(obs)