import numpy as np
from gym_minigrid.minigrid import Grid, OBJECT_TO_IDX


def render_integer_encoded_grid(
    grid_encoded, agent_direction, tile_size, att_mask=None
):
    grid_encoded = grid_encoded.copy()
    agent_mask = grid_encoded[..., 0] == 10
    agent_pos = np.flip(np.argwhere(agent_mask)[0])
    grid_encoded[agent_mask, 0] = OBJECT_TO_IDX["empty"]
    grid = Grid.decode(grid_encoded.transpose((1, 0, 2)))[0]
    return grid.render(
        tile_size,
        agent_pos,
        agent_direction,
        highlight_mask=att_mask.T if att_mask is not None else None,
    )


def render_integer_encoded_grid_path(
    grid_encoded_path, agent_direction_path, tile_size, att_masks=None
):
    att_masks = [None] * len(grid_encoded_path) if att_masks is None else att_masks
    return np.concatenate(
        [
            render_integer_encoded_grid(
                grid_encoded, agent_direction, tile_size, att_mask=att_mask
            )
            for grid_encoded, agent_direction, att_mask in zip(
                grid_encoded_path, agent_direction_path, att_masks
            )
        ],
        axis=1,
    )
