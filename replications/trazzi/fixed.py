"""Trazzi's LinkToThePastEnv with a collision-free state code and a fresh bow at episode start."""
from gym_alttp_gridworld.envs.treacherous_turn import LinkToThePastEnv

BLOCK_MASKS = 2 ** 11


class FixedLinkEnv(LinkToThePastEnv):
    def encode_state(self):
        bow_and_crystal = self.has_bow_of_light * 4 + self.encode_crystal()
        position = bow_and_crystal * self.nrow * self.ncol + self.encode_link_position()
        return position * BLOCK_MASKS + self.encode_blocks()

    def reset(self):
        self.has_bow_of_light = self.steps_since_start >= 100
        return super().reset()
