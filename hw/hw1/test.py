import matplotlib.pyplot as plt
import numpy as np

from implementation import (
    ACTION_NAMES,
    EAST,
    CalderaEnv,
    DEFAULT_PIT_PARAMS,
    DEFAULT_PIT_WEIGHTS,
    SAMPLE,
    SOUTH,
    WEST,
)


def build_lawnmower_actions(env: CalderaEnv) -> list[str]:
    actions: list[str] = []

    for row_index in range(env.num_rows):
        horizontal_action = EAST if row_index % 2 == 0 else WEST

        for col_index in range(env.num_cols):
            actions.append(SAMPLE)
            if col_index < env.num_cols - 1:
                actions.append(horizontal_action)

        if row_index < env.num_rows - 1:
            actions.append(SOUTH)

    return actions


def compute_lawnmower_energy(env: CalderaEnv) -> int:
    samples = env.num_rows * env.num_cols
    horizontal_moves = env.num_rows * (env.num_cols - 1)
    vertical_moves = env.num_rows - 1
    return samples + horizontal_moves + vertical_moves


def main_lawnmower():
    dim_x = 100
    dim_y = 100
    delta = 10
    initial_position = (0, 0)

    probe_env = CalderaEnv(
        pit_params=DEFAULT_PIT_PARAMS,
        pit_weights=DEFAULT_PIT_WEIGHTS,
        dim_x=dim_x,
        dim_y=dim_y,
        delta=delta,
        initial_position=initial_position,
    )
    max_energy = compute_lawnmower_energy(probe_env)

    caldera_env = CalderaEnv(
        pit_params=DEFAULT_PIT_PARAMS,
        pit_weights=DEFAULT_PIT_WEIGHTS,
        dim_x=dim_x,
        dim_y=dim_y,
        delta=delta,
        initial_position=initial_position,
        max_energy=max_energy,
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

    fig, _ = caldera_env.visualize_caldera()
    plt.show()


def main():
    delta = 10
    dim_x = 1000
    dim_y = 1000
    initial_position = (600, 200)

    caldera_env = CalderaEnv(
        pit_params=DEFAULT_PIT_PARAMS,
        pit_weights=DEFAULT_PIT_WEIGHTS,
        dim_x=dim_x,
        dim_y=dim_y,
        delta=delta,
        initial_position=initial_position,
    )

    current_depth = caldera_env.get_value(tuple(caldera_env.position))
    caldera_env.add_vehicle((10, 10))
    caldera_env.add_vehicle((10, 20), vehicle_size=2 * delta)
    caldera_env.add_vehicle((10, 40), vehicle_size=4 * delta)

    print(f"Current depth: {current_depth}")
    print(f"Is (10, 10) occupied? {caldera_env.is_occupied((10, 10))}")
    print(f"Is (20, 20) occupied? {caldera_env.is_occupied((20, 20))}")
    print(
        "Is the agent cell occupied by another vehicle? "
        f"{caldera_env.is_occupied(tuple(caldera_env.position), include_agent=False)}"
    )
    print(
        f"Is the agent cell occupied when including the agent? "
        f"{caldera_env.is_occupied(tuple(caldera_env.position))}"
    )

    i = 0
    path = [caldera_env.position.copy()]
    while i < 1000:
        next_action = np.random.choice(ACTION_NAMES)
        obs, reward, terminated, truncated, info = caldera_env.step(next_action)

        current_depth = caldera_env.get_value(tuple(caldera_env.position))
        path.append(caldera_env.position.copy())
        print(f"Step output: {(obs, reward, terminated, truncated, info)}")
        print(f"Agent position: {tuple(obs['position'])}")
        print(f"Current depth: {current_depth}")
        i += 1
        if terminated or truncated:
            print("Episode finished.")
            break

    fig, _ = caldera_env.visualize_caldera(agent_path=path)
    plt.show()


def main_stochastic():
    delta = 10
    dim_x = 100
    dim_y = 100
    initial_position = (60, 20)

    CalderaEnv(
        pit_params=DEFAULT_PIT_PARAMS,
        pit_weights=DEFAULT_PIT_WEIGHTS,
        dim_x=dim_x,
        dim_y=dim_y,
        delta=delta,
        initial_position=initial_position,
        stochastic=True,
        success_probabilities=(0.8, 0.8, 0.8, 0.8),
    )


if __name__ == "__main__":
    main()
