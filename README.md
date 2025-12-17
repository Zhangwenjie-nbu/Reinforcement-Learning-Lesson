# Reinforcement Learning (RL) Examples

This folder contains small Reinforcement Learning example scripts and exercises used for teaching and experiments.

Structure
---------
- `Dockerfile` - Docker build file that can be used to run the examples in a contained environment.
- `environment.yml` - Conda environment specification (if present) for installing required Python packages.
- `lesson1/` - Lesson 1 examples
  - `lesson1_bandit.py` - Multi-armed bandit example (epsilon-greedy agent).
- `lesson2/` - Lesson 2 examples
  - `2_1_markov_property.py` - Script demonstrating Markov property concepts and a simple MDP/random walk example.
- `lesson2_mdp_randomwalk.py` - Top-level lesson2 random walk / MDP demo.

Quick start
-----------
Prerequisites
- Python 3.8+ recommended
- (Optional) Conda or Miniconda for `environment.yml`

Using conda
```bash
conda env create -f environment.yml
conda activate rl
python lesson2/2_1_markov_property.py
```

Using Docker
```bash
# build image
docker build -t rl-examples .
# run the default demo
docker run --rm rl-examples python lesson2/2_1_markov_property.py
```

Running examples
----------------
- Lesson 1 (bandit):
  ```bash
  python lesson1/lesson1_bandit.py
  ```
  This runs a basic k-armed bandit simulation with an epsilon-greedy agent and prints learned action values and rewards.

- Lesson 2 (Markov / MDP):
  ```bash
  python lesson2/2_1_markov_property.py
  ```

  Lesson ...
  
  This script demonstrates the Markov property and contains a small random-walk / MDP example used in class.

Tips
----
- If you see missing dependencies, inspect `environment.yml` and install with conda or pip.
- To run multiple experiments or tune hyperparameters, wrap the scripts or edit the seed/config variables inside the scripts.

License & Notes
----------------
These examples are educational and minimal. They are good starting points for experiments and learning, not production-ready RL code.
