from typing import Any, Literal
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
from matplotlib import pyplot as plt
import numpy as np
import scipy

from connect_n.engine import _Engine, RandomEngine


class ConnectN(Env):
    def __init__(
        self,
        kernel_size: int = 4,
        board_height: int = 6,
        board_width: int = 7,
        max_episode_length: int = 200,
        render_mode: Literal["rgb_array"] | None = None,
    ):
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

    def reset(self, players_turn: int = -1, seed=None, options=None):
        self._step_counter = 0
        self._board = np.zeros((self._board_height, self._board_width), np.int16)
        self._frontier = np.zeros(self._board_width, dtype=np.uint16)
        self._players_turn = players_turn

        return self._board

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """_summary_

        Args:
            action (int): _description_

        Returns:
            tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
                - board after step
                - reward
                - terminated (reached terminal state)
                - truncated (ran out of time)
                - info
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

    def render(self):
        if self.render_mode == "rgb_array":
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

            image_flat = np.frombuffer(
                canvas.tostring_rgb(), dtype="uint8"
            )  # (H * W * 3,)
            # NOTE: reversed converts (W, H) from get_width_height to (H, W)
            image = image_flat.reshape(
                *reversed(canvas.get_width_height()), 3
            )  # (H, W, 3)
            return image

    def _check_win_condition(self) -> bool:
        for kernel in self._kernels:
            out = scipy.signal.convolve2d(
                self._board, kernel, boundary="symm", mode="valid"
            )
            if np.max(out) == self._players_turn * self._kernel_size:
                return True
        return False

    def _check_draw_condition(self) -> bool:
        return self._frontier.sum() == self._board_width * self._board_height

    def _truncated(self, current_step) -> bool:
        return current_step >= (self._max_episode_length - 1)


class ConnectNOnePlayer(Env):
    def __init__(
        self,
        kernel_size: int = 4,
        board_height: int = 6,
        board_width: int = 7,
        max_episode_length: int = 200,  # how often the player can call step
        render_mode: Literal["rgb_array"] | None = None,
        engine: _Engine | None = RandomEngine,
    ):
        super().__init__()
        self._env = ConnectN(
            kernel_size,
            board_height,
            board_width,
            2 * max_episode_length,  # two times because one player is the engine
            render_mode,
        )
        self._engine = engine(self._env)

    def render(self):
        return self._env.render()

    def reset(
        self,
        first_player: int = 1,  # -1 indication engine, 1 indication foreign player
        seed=None,
        options=None,
    ):
        state = self._env.reset(players_turn=first_player, seed=seed, options=options)

        if first_player == -1:  # the engine has to play first
            state, _, _, _, _ = self._env.step(self._engine.get_action(state))

        return state

    def step(self, action: int):
        # player plays
        state, reward, terminated, truncated, info = self._env.step(action)
        if terminated or truncated:
            return state, reward, terminated, truncated, info
        # engine plays
        state, _, terminated, truncated, _ = self._env.step(self._engine.get_action(state))
        return state, reward, terminated, truncated, info

