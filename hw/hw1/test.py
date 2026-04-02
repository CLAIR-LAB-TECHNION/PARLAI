import matplotlib.pyplot as plt
import numpy as np

from utils import BivariateNormalStruct
from caldera_env import (
    ACTION_NAMES,
    MOVE_EAST,
    MOVE_SOUTH,
    MOVE_WEST,
    DEFAULT_PIT_PARAMS,
    DEFAULT_PIT_WEIGHTS,
    SAMPLE,
)
from to_implement import CalderaEnv, POCalderaEnv, SCalderaEnv, stochastic_effet_wrong_turn


def build_lawnmower_actions(env: CalderaEnv) -> list[str]:
    actions: list[str] = []
    num_cols = len(env.x_grid_coords)
    num_rows = len(env.y_grid_coords)

    for row_index in range(num_rows):
        horizontal_action = MOVE_EAST if row_index % 2 == 0 else MOVE_WEST

        for col_index in range(num_cols):
            actions.append(SAMPLE)
            if col_index < num_cols - 1:
                actions.append(horizontal_action)

        if row_index < num_rows - 1:
            actions.append(MOVE_SOUTH)

    return actions


def compute_lawnmower_energy(env: CalderaEnv) -> int:
    num_cols = len(env.x_grid_coords)
    num_rows = len(env.y_grid_coords)
    samples = num_rows * num_cols
    horizontal_moves = num_rows * (num_cols - 1)
    vertical_moves = num_rows - 1
    return samples + horizontal_moves + vertical_moves


def main_lawnmower():
    dim_x = 100
    dim_y = 100
    sampling_res = 10
    initial_position = (0, 0)

    probe_env = CalderaEnv(
        dim_x=dim_x,
        dim_y=dim_y,
        pit_params=DEFAULT_PIT_PARAMS,
        pit_weights=DEFAULT_PIT_WEIGHTS,
        sampling_res=sampling_res,
        initial_position=initial_position,
    )
    initial_energy = compute_lawnmower_energy(probe_env)

    caldera_env = CalderaEnv(
        dim_x=dim_x,
        dim_y=dim_y,
        pit_params=DEFAULT_PIT_PARAMS,
        pit_weights=DEFAULT_PIT_WEIGHTS,
        sampling_res=sampling_res,
        initial_position=initial_position,
        initial_energy=initial_energy,
    )

    actions = build_lawnmower_actions(caldera_env)
    total_reward = 0.0
    counter = 0

    for step_index, action in enumerate(actions, start=1):
        counter += 1
        if counter < 100:
            obs, reward, terminated, truncated, _ = caldera_env.step(action)
            total_reward += reward

        print(
            f"Step {step_index}: action={action}, position={tuple(obs['position'])}, "
            f"reward={reward:.2f}, energy={obs['energy']}"
        )

        if terminated or truncated:
            print("Episode finished before the full lawn-mowing pattern completed.")
            break

    print(f"Visited cells: {len(caldera_env.sampled_cells)}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Final position: {tuple(caldera_env.position)}")

    fig, _ = caldera_env.visualize()
    plt.show()


def main():
    sampling_res = 10
    dim_x = 1000
    dim_y = 1000
    initial_position = (10, 45)
    other_vehicles = [
        ((50, 200), 8 * sampling_res),
        ((200, 200), 8 * sampling_res),
        ((300, 150), 8 * sampling_res),
        ((100, 150), 8 * sampling_res),
        ((200, 400), 8 * sampling_res),
    ]

    
    # Default parameters for the three pits in the caldera, along with their weights that control how deep they are.
    #PIT_PARAMS = (
    #BivariateNormalStruct(x=0.0, y=0.0, sigmax=0.16, sigmay=0.15, mux=0.2, muy=0.2),
    #BivariateNormalStruct(x=0.0, y=0.0, sigmax=0.18, sigmay=0.2, mux=0.5, muy=0.7),
    #BivariateNormalStruct(x=0.0, y=0.0, sigmax=0.17, sigmay=0.15, mux=0.6, muy=0.3),
    #BivariateNormalStruct(x=0.0, y=0.0, sigmax=0.2, sigmay=0.2, mux=0.9, muy=0.1),
    #)
    #PIT_WEIGHTS = (16000.0, 22000.0, 18000.0, 50000.0)

    caldera_env = CalderaEnv(
        dim_x=dim_x,
        dim_y=dim_y,
        sampling_res=sampling_res,
        movement_size=sampling_res,
        initial_position=initial_position,
        initial_energy=1000,
        other_vehicles=other_vehicles,
    )

    current_depth = caldera_env._get_sample()

    print(f"Current depth: {current_depth[1]:.2f}")
    print(f"Is (100, 10) occupied? {caldera_env.is_occupied((10, 10))}")
    print(f"Is (300, 20) occupied? {caldera_env.is_occupied((30, 20))}")
    print(
        "Is the agent cell occupied by another vehicle? "
        f"{caldera_env.is_occupied(tuple(caldera_env.position), include_agent=False)}"
    )
    print(
        f"Is the agent cell occupied when including the agent? "
        f"{caldera_env.is_occupied(tuple(caldera_env.position),include_agent=True)}"
    )
    

    i = 0
    path = [caldera_env.position.copy()]
    while i < 1000:
        next_action = np.random.choice(ACTION_NAMES)
        obs, reward, terminated, truncated, info = caldera_env.step(next_action)

        current_depth = caldera_env._get_sample()
        path.append(caldera_env.position.copy())
        print(f"terminated is: {terminated}")
        print(f"Agent position: {tuple(obs['position'])}")
        print(f"Current depth: {current_depth[1]:.2f}")
        i += 1
        if terminated or truncated:
            print("Episode finished.")
            break

    fig, _ = caldera_env.visualize(agent_path=path,show_grid_lines=True)
    plt.show()


def main_stochastic():
    sampling_res = 10
    dim_x = 100
    dim_y = 100
    initial_position = (60, 20)

    caldera_env = SCalderaEnv(
        dim_x=dim_x,
        dim_y=dim_y,
        pit_params=DEFAULT_PIT_PARAMS,
        pit_weights=DEFAULT_PIT_WEIGHTS,
        sampling_res=sampling_res,
        initial_position=initial_position,
        stochastic_effet_function=stochastic_effet_wrong_turn,
    )
    i = 0
    path = [caldera_env.position.copy()]
    while i < 1000:
        next_action = np.random.choice(ACTION_NAMES)
        obs, reward, terminated, truncated, info = caldera_env.step(next_action)

        current_depth = caldera_env._get_sample()
        path.append(caldera_env.position.copy())
        print(f"Step output: {(obs, reward, terminated, truncated, info)}")
        print(f"Agent position: {tuple(obs['position'])}")
        print(f"Current depth: {current_depth[1]:.2f}")
        i += 1
        if terminated or truncated:
            print("Episode finished.")
            break

    fig, _ = caldera_env.visualize(agent_path=path,show_grid_lines=True)
    plt.show()


def main_test_all():
    sampling_res = 5
    dim_x = 100
    dim_y = 100
    initial_position = (66, 10)
    other_vehicles = [
        ((22, 83), 15),
        ((84, 15), 11),
        ((40, 30), 30),
    ]


    
    # Default parameters for the three pits in the caldera, along with their weights that control how deep they are.
    #PIT_PARAMS = (
    #BivariateNormalStruct(x=0.0, y=0.0, sigmax=0.16, sigmay=0.15, mux=0.2, muy=0.2),
    #BivariateNormalStruct(x=0.0, y=0.0, sigmax=0.18, sigmay=0.2, mux=0.5, muy=0.7),
    #BivariateNormalStruct(x=0.0, y=0.0, sigmax=0.17, sigmay=0.15, mux=0.6, muy=0.3),
    #BivariateNormalStruct(x=0.0, y=0.0, sigmax=0.2, sigmay=0.2, mux=0.9, muy=0.1),
    #)
    #PIT_WEIGHTS = (16000.0, 22000.0, 18000.0, 50000.0)

    caldera_env = POCalderaEnv(
        dim_x=dim_x,
        dim_y=dim_y,
        sampling_res=sampling_res,
        initial_position=initial_position,
        movement_size=1,
        observability_distance=5,
        other_vehicles=other_vehicles,
        end_episode_on_collision=True,

    )
    current_depth = caldera_env._get_sample()

    print(f"Current depth: {current_depth[1]:.2f}")
    #print(f"Is (100, 10) occupied? {caldera_env.is_occupied((10, 10))}")
    #print(f"Is (300, 20) occupied? {caldera_env.is_occupied((30, 20))}")
    print(
        "Is the agent cell occupied by another vehicle? "
        f"{caldera_env.is_occupied(tuple(caldera_env.position), include_agent=False)}"
    )
    print(
        f"Is the agent cell occupied when including the agent? "
        f"{caldera_env.is_occupied(tuple(caldera_env.position),include_agent=True)}"
    )  


    # move north 
    obs, reward, terminated, truncated, info = caldera_env.step("MOVE_NORTH")
    print(f"Step output: {(obs, reward, terminated, truncated, info)}")    
    print(f"Agent position: {tuple(obs['position'])}")
    print(f"Current depth: {caldera_env._get_sample()[1]:.2f}")

    # move east 
    number_of_east_moves = 10
    for i in range(number_of_east_moves):
        print(f"Moving east=step {i}")
        obs, reward, terminated, truncated, info = caldera_env.step("MOVE_EAST")
        print(f"Step output: {(obs, reward, terminated, truncated, info)}")    
        print(f"Agent position: {tuple(obs['position'])}")
        print(f"Current depth: {caldera_env._get_sample()[1]:.2f}")
    # move north 
    number_of_north_moves = 1000
    for i in range(number_of_north_moves):
        print(f"Moving north=step {i}")
        obs, reward, terminated, truncated, info = caldera_env.step("MOVE_NORTH")
        print(f"Step output: {(obs, reward, terminated, truncated, info)}")    
        print(f"terminated is: {terminated}")
        print(f"Agent position: {tuple(obs['position'])}")
        print(f"Current depth: {caldera_env._get_sample()[1]:.2f}")
   
    # move west 
    number_of_west_moves = 10
    for i in range(number_of_west_moves):
        print(f"Moving west=step {i}")
        obs, reward, terminated, truncated, info = caldera_env.step("MOVE_WEST")
        print(f"Step output: {(obs, reward, terminated, truncated, info)}")    
        print(f"Agent position: {tuple(obs['position'])}")
        print(f"Current depth: {caldera_env._get_sample()[1]:.2f}")
   
    i = 10
    path = [caldera_env.position.copy()]
    while i < -1:
        next_action = np.random.choice(ACTION_NAMES)
        obs, reward, terminated, truncated, info = caldera_env.step(next_action)

        current_depth = caldera_env._get_sample()
        path.append(caldera_env.position.copy())
        print(f"Step output: {(obs, reward, terminated, truncated, info)}")
        print(f"Agent position: {tuple(obs['position'])}")
        print(f"Current depth: {current_depth[1]:.2f}")
        i += 1
        if terminated or truncated:
            print("Episode finished.")
            break
    fig, _ = caldera_env.visualize(show_grid_lines=True,show_agent_path=True)
    ##fig, _ = caldera_env.visualize_caldera(agent_path=path,show_grid_lines=True)
    plt.show()



def visualize_default_caldera_with_grid_lines():
    caldera_env = CalderaEnv(initial_position=(60, 20), sampling_res=10)

    fig, _ = caldera_env.visualize(show_grid_lines=True)
    plt.show()
 
 

if __name__ == "__main__":
    main_stochastic()
    main_test_all()    
    #visualize_default_caldera_with_grid_lines()
    #main_visual()
