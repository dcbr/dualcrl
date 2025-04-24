# Dual Constrained Reinforcement Learning
This repository provides the necessary code to reproduce all experiments of the paper *"A Dual Perspective of Reinforcement Learning for Imposing Policy Constraints"* [1].

An example [Pytorch](https://pytorch.org) implementation of the <tt>DualCRL</tt> algorithm is provided, with support for any combination of entropy regularization, value constraints, density constraints and transition constraints.
Custom constrained setups are defined for the `CliffWalking-v0` and `Pendulum-v1` [Gymnasium environments](https://gymnasium.farama.org).
Further details are provided in the paper's Experiments section.

<table>
<tr>
<td>

https://github.com/user-attachments/assets/c3c32c2c-3ba0-4098-9580-9a8216c3246e

</td>
<td>

https://github.com/user-attachments/assets/94a3130d-e4f7-4458-91b1-43e7442f4226

</td>
</tr>
</table>

## Installation
1. Clone this repository.

   ``git clone https://github.com/dcbr/dualcrl``

   ``cd dualcrl``

2. Install the required packages, as specified in `environment.yml`.
   This can be easily done by creating a virtual environment (using e.g. conda or venv).

   ``conda env create -f environment.yml``

## Usage
Activate the virtual environment, using e.g. `conda activate dualcrl`. Afterwards, you can simply run the `main` script with suitable arguments to train the models or analyze their performance.
For example

``python main.py --mode train --job cliffwalk``

to train on the cliff walking environment (with additional policy constraints).

To reproduce all results of Section VI, first train on all jobs with ``python main.py --mode train --job [JOB] --uid paper``, followed by the analysis ``python main.py --mode analyze --job [JOB] --uid paper``. Beware that this might take a while to complete, depending on your hardware!

A summary of the most relevant parameters to this script is provided below.
Check ``python main.py --help`` for a full overview of supported parameters.

| Parameter | Supported values        | Description                                                                   |
|:----------|:------------------------|:------------------------------------------------------------------------------|
| `--mode`  | `train`, `analyze`      | Run mode. Either train models, or analyze (and summarize) the results.        |
| `--job`   | `cliffwalk`, `pendulum` | Job to run. The job file defines the environment and constraints to train on. |
| `--uid`   | Any value               | Unique identifier for a job run.                                              |

## References
[1] De Cooman, B., Suykens, J.: A Dual Perspective of Reinforcement Learning for Imposing Policy Constraints. Accepted for publication in *IEEE Transactions on Artificial Intelligence*.
