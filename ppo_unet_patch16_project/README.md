# PPO U-Net Patch 16 Project

This project implements a reinforcement learning model using a Proximal Policy Optimization (PPO) algorithm combined with a U-Net architecture for image segmentation tasks. The model operates on image patches and applies various actions to improve segmentation performance.

## Project Structure

```
ppo_unet_patch16_project
├── src
│   ├── __init__.py
│   ├── main.py                 # Entry point to run training / evaluation
│   ├── train.py                # Orchestrates the training process
│   ├── models
│   │   ├── __init__.py
│   │   └── actor_critic_unet.py # Implementation of the ActorCriticUNet class
│   ├── env
│   │   ├── __init__.py
│   │   ├── simulator.py        # Functions for environment interaction
│   │   └── actions.py          # Functions for applying actions to patches
│   ├── utils
│   │   ├── __init__.py
│   │   ├── io.py               # Utility functions for I/O operations
│   │   ├── patches.py          # Functions for handling image patches
│   │   ├── metrics.py          # Functions for computing evaluation metrics
│   │   └── ppo.py              # PPO configuration and update functions
│   └── data
│       ├── __init__.py
│       └── dataset_stub.py     # Placeholder for DataLoader
├── tests
│   ├── test_patches.py         # Unit tests for patch-related functions
│   ├── test_actions.py         # Unit tests for action-related functions
│   └── test_ppo.py             # Unit tests for PPO-related functions
├── requirements.txt             # Project dependencies
├── setup.cfg                    # Configuration settings
├── pyproject.toml               # Project metadata and dependency management
└── README.md                    # Project documentation
```

## Installation

To install the required dependencies, run:

```
pip install -r requirements.txt
```

## Usage

To train the model, run the following command:

```
python src/main.py
```

This will start the training process using the defined configurations and data loading mechanisms.

## Testing

To run the tests, use:

```
pytest tests/
```

This will execute all unit tests defined in the `tests` directory.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.