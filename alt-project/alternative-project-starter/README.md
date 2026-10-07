# PARLAI Spring 2026, Alternative Project, starter package

This folder has everything you need for the project. You do not need any files from the homework.

## What is in this folder

| File | What it is |
|---|---|
| `mission.py` | Builds your mission and its environment. Do not change it. |
| `environment.yml` | The conda environment. We run your code with it on lambda. |
| `py-sbatch.sh` | Runs a Python script on a lambda compute node through slurm. |
| `check_setup.py` | Checks that your setup works and draws your mission. |
| `part1_currents.py` | Your code for Part 1. |
| `part2_planning.py` | Your code for Part 2. |
| `part3_learning.py` | Your code for Part 3. |
| `part4_summary.py` | Your code for the summary table of Part 4. |
| `pddl/domain.pddl` | Your PDDL domain for 2.1. |
| `results/` | Save the raw results (CSV or JSON) of every table and figure here. |
| `figures/` | Save your figures here. |
| `ai_use.txt` | Fill in which AI tools you used and for what. |

The part files contain function names and short descriptions to help you start. You may add functions and files. Each part file must keep running with `python partX_....py --id <your ID>`.

## Setup on the lambda server

We will test your code on lambda, so set it up there first.

1. Connect to lambda with `ssh <user>@lambda.cs.technion.ac.il`. From home you need the Technion VPN.
2. If you do not have conda on lambda yet, install Miniconda in your home folder.
   ```
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
   source $HOME/miniconda3/etc/profile.d/conda.sh
   ```
3. Copy this folder to lambda, for example with `rsync -rvz alternative-project-starter <user>@lambda.cs.technion.ac.il:~/`
4. Create the environment from inside the folder with `conda env create -f environment.yml`
5. Run the check on a compute node.
   ```
   chmod +x py-sbatch.sh
   ./py-sbatch.sh check_setup.py --id <your ID>
   ```
   Then read the output with `cat slurm-<job id>.out`. The last line should say `Setup OK`.

Do not run computations on the lambda gateway itself. Always use `py-sbatch.sh`, `sbatch` or `srun`.

You may also work on your own computer with the same `environment.yml`. Before you submit, run every part file on lambda once more.

## Using mission.py

```python
from mission import make_mission, describe_mission, evaluate, run_episode, plot_mission, plot_path

env, info = make_mission(123456789)      # use your own ID
print(describe_mission(info))            # put this at the start of your report

obs, step_info = env.reset(seed=0)
# obs = {"cell": array([x, y]), "energy": e, "collected": array([0, 0, 0])}
# step_info["action_mask"] marks the available actions, in the order of mission.ACTIONS
obs, reward, done, _, step_info = env.step("MOVE_NORTH")

state = env.get_state()                  # ((x, y), energy, (c0, c1, c2))
env.set_state((3, 4), 20, (1, 0, 0))     # only where the project allows it
```

A policy is any function `policy(obs, step_info)` that returns an action name or index. `evaluate(policy, env, n_episodes, first_seed)` runs episodes with seeds `first_seed`, `first_seed + 1`, and so on, and returns the mean return, its standard error, the full-success rate, the loss rate and the result of every episode. If your policy object has a `reset()` method, `evaluate` calls it at the start of every episode.

`run_episode(policy, env, seed)` returns a record of one episode. Draw it with `plot_path(info, episode)`. `plot_arrows(info, actions_by_cell)` draws one action per cell, which is useful for 2.3(d).

## Submitting

Follow Section "What to submit" in the project document. Include this whole folder with your code, `results/`, `figures/`, your report and `ai_use.txt`.
