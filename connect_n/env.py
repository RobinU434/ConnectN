from typing import Any, Literal
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import scipy

from connect_n.engine import _Engine, RandomEngine


class ConnectN(Env):
    """
    ConnectN is a custom Gymnasium environment that simulates the Connect-N game,
    where two players alternate turns to drop their pieces into a grid and aim to connect
    N pieces either vertically, horizontally, or diagonally.

    Attributes:
        kernel_size (int): Number of consecutive pieces required to win (default: 4).
        board_height (int): The number of rows in the board (default: 6).
        board_width (int): The number of columns in the board (default: 7).
        max_episode_length (int): The maximum number of steps allowed before truncation (default: 200).
        render_mode (str or None): If "rgb_array", renders the board as an RGB array.
    """

    def __init__(
        self,
        kernel_size: int = 4,
        board_height: int = 6,
        board_width: int = 7,
        max_episode_length: int = 200,
        render_mode: Literal["rgb_array", "human"] | None = None,
    ):
        """
        Initializes the ConnectN environment with the specified board size, win condition, and other settings.

        Args:
            kernel_size (int): The number of pieces a player must connect to win.
            board_height (int): The height of the board.
            board_width (int): The width of the board.
            max_episode_length (int): Maximum number of steps allowed before truncation.
            render_mode (Literal["rgb_array", "human"] | None): Render mode for visualizing the board.
        """
        super().__init__()

        self._board_height = board_height
        self._board_width = board_width
        self._max_episode_length = max_episode_length
        self._kernel_size = kernel_size
        self._win_reward = 1
        self._draw_reward = 0

        self._step_counter: int
        self._board: np.ndarray
        self._frontier: np.ndarray
        self._players_turn: np.ndarray
        self.reset()

        self.render_mode = render_mode
        self.observation_space = Box(-1, 1, self._board.shape)
        self.action_space = Discrete(self._board_width)

        self._kernels = self._create_kernels(kernel_size=self._kernel_size)

    @staticmethod
    def _create_kernels(kernel_size: int) -> np.ndarray:
        """
        Creates kernels used to detect winning patterns on the board, both horizontally,
        vertically, and diagonally.

        Args:
            kernel_size (int): The size of the winning sequence.

        Returns:
            np.ndarray: A stacked array of kernels used for detecting win conditions.
        """
        kernels = []
        for i in range(kernel_size):
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[i] = 1
            # add horizontal
            kernels.append(kernel)
            # add vertical
            kernels.append(kernel.T)

        # add diagonals
        kernels.append(np.eye(kernel_size))
        kernels.append(np.eye(kernel_size)[::-1])
        kernels = np.stack(kernels)
        return kernels

    def reset(self, players_turn: int = -1, seed=None, options=None) -> np.ndarray:
        """
        Resets the environment to its initial state.

        Args:
            players_turn (int): Specifies which player goes first. -1 for Player 1, 1 for Player 2 (default: -1).
            seed: Seed for the environment's random number generator.
            options: Additional options for reset.

        Returns:
            np.ndarray: The initial board state.
        """
        self._step_counter = 0
        self._board = np.zeros((self._board_height, self._board_width), np.int16)
        self._frontier = np.zeros(self._board_width, dtype=np.uint16)
        self._players_turn = players_turn

        return self._board

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Executes a game step where the current player places their piece in the specified column.

        Args:
            action (int): The column (0 to board_width-1) where the player drops their piece.

        Returns:
            tuple: A tuple containing:
                - state (np.ndarray): The updated board state.
                - reward (float): The reward for the action (1 for win, 0 for ongoing, etc.).
                - terminated (bool): Whether the game has ended (win or draw).
                - truncated (bool): Whether the episode was truncated due to reaching max steps.
                - info (dict): Additional information about the step.
        """
        assert self.action_space.contains(action)

        self._step_counter += 1
        truncated = self._truncated(self._step_counter)

        frontier = self._frontier[action]
        # NOTE: if the agent chooses an action which is not available return the same with zero reward
        if frontier == self._board_height:
            state = self._board
            reward = 0
            terminated = False
            return state, reward, terminated, truncated, {}

        # place the stone
        self._board[frontier, action] = self._players_turn
        self._frontier[action] += 1
        terminated = False
        reward = 0
        # check if somebody has won
        if self._check_win_condition():
            reward = self._win_reward
            terminated = True
        elif self._check_draw_condition():
            reward = self._draw_reward
            terminated = True

        # change player
        self._players_turn *= -1

        # TODO: check if it is possible somebody wins and truncate early
        return self._board, reward, terminated, truncated, {}

    def render(self) -> tuple[Figure, Axes] | np.ndarray | None :
        """
        Renders the current game state as an RGB array if `render_mode` is "rgb_array".

        Returns:
            tuple[Figure, Axes] | np.ndarray: A rendered RGB image of the current board state.
        """
        if self.render_mode is not None:
            fig = plt.figure()
            canvas = fig.canvas
            plt.imshow(
                self._board[::-1],
                interpolation="none",
                vmin=-1,
                vmax=1,
                aspect="equal",
                cmap="bwr",
            )
            ax = plt.gca()

            # Major ticks
            ax.set_xticks(np.arange(0, self._board_width, 1))
            ax.set_yticks(np.arange(0, self._board_height, 1))

            # Labels for major ticks
            ax.set_xticklabels(np.arange(1, self._board_width + 1, 1))
            ax.set_yticklabels(np.arange(1, self._board_height + 1, 1))

            # Minor ticks
            ax.set_xticks(np.arange(-0.5, self._board_width, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, self._board_height, 1), minor=True)

            # Gridlines based on minor ticks
            ax.grid(which="minor", color="grey", linestyle="-", linewidth=2)

            # Remove minor ticks
            ax.tick_params(which="minor", bottom=False, left=False)

            canvas.draw()  # Draw the canvas, cache the renderer

        if self.render_mode == "human":
            return fig, ax

        if self.render_mode == "rgb_array":
            image_flat = np.frombuffer(
                canvas.tostring_rgb(), dtype="uint8"
            )  # (H * W * 3,)
            # NOTE: reversed converts (W, H) from get_width_height to (H, W)
            image = image_flat.reshape(
                *reversed(canvas.get_width_height()), 3
            )  # (H, W, 3)
            return image
        return None

    def _check_win_condition(self) -> bool:
        """
        Checks whether the current player has won the game by connecting the required number of pieces.

        Returns:
            bool: True if the current player has won, False otherwise.
        """
        for kernel in self._kernels:
            out = scipy.signal.convolve2d(
                self._board, kernel, boundary="symm", mode="valid"
            )
            if np.max(out) == self._players_turn * self._kernel_size:
                return True
        return False

    def _check_draw_condition(self) -> bool:
        """
        Checks whether the game has ended in a draw (i.e., the board is full).

        Returns:
            bool: True if the game is a draw, False otherwise.
        """
        return self._frontier.sum() == self._board_width * self._board_height

    def _truncated(self, current_step) -> bool:
        """
        Checks whether the game should be truncated based on the current step count.

        Args:
            current_step (int): The current step number.

        Returns:
            bool: True if the episode should be truncated, False otherwise.
        """
        return current_step >= (self._max_episode_length - 1)
    
    def get_step_counter(self) -> int:
        """get step counter

        Returns:
            int: current step index
        """
        return self._step_counter


class ConnectNOnePlayer(Env):
    """
    ConnectNOnePlayer is a single-player variant of the ConnectN environment, where the player plays
    against an AI opponent (engine). The AI's behavior can be controlled by specifying a custom engine.

    Attributes:
        kernel_size (int): Number of consecutive pieces required to win (default: 4).
        board_height (int): The number of rows in the board (default: 6).
        board_width (int): The number of columns in the board (default: 7).
        max_episode_length (int): The maximum number of steps allowed before truncation (default: 200).
        render_mode (str or None): If "rgb_array", renders the board as an RGB array.
        engine (_Engine): The AI engine that controls the opponent's moves (default: RandomEngine).
    """

    def __init__(
        self,
        kernel_size: int = 4,
        board_height: int = 6,
        board_width: int = 7,
        max_episode_length: int = 200,  # how often the player can call step
        render_mode: Literal["rgb_array"] | None = None,
        engine: _Engine | None = RandomEngine,
    ):
        """
        Initializes the ConnectNOnePlayer environment, where the player competes against an AI engine.

        Args:
            kernel_size (int): The number of pieces required to win the game.
            board_height (int): The number of rows in the board.
            board_width (int): The number of columns in the board.
            max_episode_length (int): Maximum number of steps allowed before truncation.
            render_mode (Literal["rgb_array"] | None): Render mode for visualizing the board.
            engine (_Engine | None): The engine controlling the AI opponent (default: RandomEngine).
        """
        super().__init__()
        self._env = ConnectN(
            kernel_size,
            board_height,
            board_width,
            2 * max_episode_length,  # two times because one player is the engine
            render_mode,
        )
        self._engine = engine(self._env)

        self.render_mode = render_mode
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space

    def render(self) -> np.ndarray:
        """
        Renders the current state of the game using the ConnectN environment's render method.

        Returns:
            np.ndarray: A rendered RGB image of the current board state.
        """
        return self._env.render()

    def reset(
        self,
        first_player: int = 1,  # -1 indication engine, 1 indication foreign player
        seed=None,
        options=None,
    ) -> np.ndarray:
        """
        Resets the game state and determines whether the player or the AI goes first.

        Args:
            first_player (int): Specifies which player moves first. -1 if the AI starts, 1 if the human starts (default: 1).
            seed: Seed for the environment's random number generator.
            options: Additional options for reset.

        Returns:
            np.ndarray: The initial board state, possibly after the AI's first move.
        """
        state = self._env.reset(players_turn=first_player, seed=seed, options=options)

        if first_player == -1:  # the engine has to play first
            state, _, _, _, _ = self._env.step(self._engine.get_action(state))

        return state

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Takes a step in the game. First, the player's move is executed, and then the AI opponent makes a move.

        Args:
            action (int): The column (0 to board_width-1) where the player drops their piece.

        Returns:
            tuple: A tuple containing:
                - state (np.ndarray): The updated board state.
                - reward (float): The reward for the player's move.
                - terminated (bool): Whether the game has ended (win or draw).
                - truncated (bool): Whether the episode was truncated due to reaching max steps.
                - info (dict): Additional information about the step.
        """
        # player plays
        state, reward, terminated, truncated, info = self._env.step(action)
        if terminated or truncated:
            return state, reward, terminated, truncated, info
        # engine plays
        state, _, terminated, truncated, _ = self._env.step(
            self._engine.get_action(state)
        )
        return state, reward, terminated, truncated, info

    def get_step_counter(self) -> int:
        """get step counter

        Returns:
            int: current step index
        """
        return self._env.get_step_counter() // 2