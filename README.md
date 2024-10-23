# ConnectN Gymnasium Environment

## Overview

This repository provides a custom implementation of a **ConnectN** environment using the Gymnasium API (an upgraded version of OpenAI Gym). It offers a standard two-player version of ConnectN, as well as a one-player version where the opponent is controlled by an AI engine. The environment is designed to be flexible, allowing for customization of board size, winning kernel size, and step limits.

## Environment Features

### ConnectN Environment
The `ConnectN` class represents the standard version of the ConnectN game. The game works on a 2D grid where two players take turns placing pieces, aiming to connect a specified number of pieces either vertically, horizontally, or diagonally.

Key Parameters:
- **kernel_size**: Specifies the number of consecutive pieces required to win (default: 4, as in Connect4).
- **board_height**: The number of rows on the board (default: 6).
- **board_width**: The number of columns on the board (default: 7).
- **max_episode_length**: Maximum number of steps allowed before the episode is truncated (default: 200).
- **render_mode**: If set to "rgb_array", the environment can render the game board as a NumPy array for visualization purposes.

### ConnectNOnePlayer Environment
The `ConnectNOnePlayer` class allows a single player to play against an AI opponent. The AI uses a customizable engine (default: random moves) to play.

Key Parameters:
- **engine**: A custom or prebuilt engine that defines the opponent's strategy. By default, it is set to `RandomEngine`.

## Observation Space and Action Space

### Observation Space:
- The observation space is a 2D grid representing the game board.
- The grid contains values of `-1`, `0`, and `1`:
  - `-1`: The piece belongs to the second player.
  - `0`: The cell is empty.
  - `1`: The piece belongs to the first player.
- The size of the grid is determined by `board_height` and `board_width`.

### Action Space:
- The action space is a discrete space with values ranging from `0` to `board_width - 1`. Each action corresponds to a column where the player wants to drop their piece.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install Gymnasium if you haven't already:
   ```bash
   pip install gymnasium
   ```

## Usage

### Running the Environment

#### Example Usage (Two-Player Environment):

```python
from connect_n.env import ConnectN

# Initialize environment
env = ConnectN(render_mode="rgb_array")

# Reset the environment to start a new game
state = env.reset()

done = False
while not done:
    action = env.action_space.sample()  # Choose a random action
    state, reward, terminated, truncated, info = env.step(action)  # Take a step
    done = terminated or truncated
    env.render()  # Optionally render the board
```

#### Example Usage (One-Player Environment):

```python
from connect_n.env import ConnectNOnePlayer

# Initialize one-player environment
env = ConnectNOnePlayer(render_mode="rgb_array")

# Reset the environment with the player going first
state = env.reset(first_player=1)

done = False
while not done:
    action = env.action_space.sample()  # Choose a random action
    state, reward, terminated, truncated, info = env.step(action)  # Take a step
    done = terminated or truncated
    env.render()  # Optionally render the board
```

### Rendering
To render the board visually, set the `render_mode` to `"rgb_array"` in the environment initialization:
```python
env = ConnectN(render_mode="rgb_array")
```
Calling `env.render()` will return an image representation of the current board state.

## API Documentation

### ConnectN

- **reset**:
  ```python
  def reset(self, players_turn: int = -1, seed=None, options=None) -> np.ndarray
  ```
  Resets the environment to its initial state. Can specify which player starts first.

- **step**:
  ```python
  def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]
  ```
  Takes an action (column) and returns the next state of the game, reward, and whether the game has ended.

- **render**:
  ```python
  def render(self) -> np.ndarray
  ```
  Renders the current state of the game board if `render_mode` is set to `"rgb_array"`.

### ConnectNOnePlayer

- **reset**:
  ```python
  def reset(self, first_player: int = 1, seed=None, options=None) -> np.ndarray
  ```
  Resets the environment with the option to specify if the AI or player moves first.

- **step**:
  ```python
  def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]
  ```
  Takes the player's action, followed by the AI's action, and returns the updated state.

## Customizing the AI Engine

The `ConnectNOnePlayer` class uses an engine to control the AI's actions. You can provide your own AI engine by subclassing the `_Engine` class and passing it into the environment.

## Contributing

Feel free to fork this repository, create new features, and open a pull request! All contributions are welcome.

## License

This project is licensed under the MIT License.
