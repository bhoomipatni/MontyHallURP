# MontyHallURP
Undergraduate Research Project on the Monty Hall Problem

## Description
This project implements a Q-learning reinforcement learning agent that learns to play the Monty Hall problem in two variants:
- **Classic Monty Hall**: The standard problem where switching doors increases winning probability
- **Evil Monty Hall**: A variant where the host only offers a switch when the contestant initially picks the prize

## Setup

### Requirements
- Python 3.x
- matplotlib

### Installation
1. Clone the repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage
Run the simulation:
```bash
python3 RL_EvilClassic.py
```

This will:
- Train Q-learning agents on both the classic and evil Monty Hall variants
- Print the learned Q-values for each strategy
- Generate a plot (`monty_winrate.png`) showing the learning progress over episodes

## Results
- In the **classic** variant, the agent learns that switching (Q[1]) is better than staying (Q[0])
- In the **evil** variant, the agent learns that staying (Q[0]) is optimal, as the host manipulates the switching option
