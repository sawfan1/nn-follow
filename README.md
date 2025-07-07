# Taxi AI - Q-Learning Project

A simple taxi AI using Q-Learning to solve the OpenAI Gym Taxi environment. The agent learns to efficiently pick up and drop off passengers in a 5x5 grid world.

## Overview

The taxi navigates a grid to transport passengers while maximizing rewards (+20 for successful dropoff, -10 for illegal actions, -1 per step). Uses Q-Learning with epsilon-greedy exploration to learn optimal policies.

## Requirements & Installation

```bash
pip install gym numpy
python train_taxi_ai.py
```

## Results

After training, the agent achieves near-optimal performance with efficient pathfinding and successful passenger transport.

## Acknowledgments

Tutorial followed from **NeuralNine YouTube Channel** for educational purposes.
