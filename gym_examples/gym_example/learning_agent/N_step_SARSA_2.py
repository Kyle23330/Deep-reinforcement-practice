import numpy as np
from gym_example.learning_agent.iht import tiles
from gym_example.learning_agent.iht import IHT


class MountainCarTileCoder:
    def __init__(self, iht_size=4096, num_tilings=8, num_tiles=8):
        """
        Initializes the MountainCar Tile Coder
        Initializers:
        iht_size -- int, the size of the index hash table, typically a power of 2
        num_tilings -- int, the number of tilings
        num_tiles -- int, the number of tiles. Here both the width and height of the
                     tile coder are the same
        Class Variables:
        self.iht -- tc.IHT, the index hash table that the tile coder will use
        self.num_tilings -- int, the number of tilings the tile coder will use
        self.num_tiles -- int, the number of tiles the tile coder will use
        """
        self.iht = IHT(iht_size)
        self.num_tilings = num_tilings
        self.num_tiles = num_tiles
    
    def get_tiles(self, position, velocity):
        """
        Takes in a position and velocity from the mountaincar environment
        and returns a numpy array of active tiles.
        
        Arguments:
        position -- float, the position of the agent between -1.2 and 0.5
        velocity -- float, the velocity of the agent between -0.07 and 0.07
        returns:
        tiles - np.array, active tiles
        """
        # Set the max and min of position and velocity to scale the input
        # POSITION_MIN
        # POSITION_MAX
        # VELOCITY_MIN
        # VELOCITY_MAX
        ### START CODE HERE ###
        position_max = 0.5
        position_min = -1.2
        velocity_max = 0.07
        velocity_min = -0.07
        ### END CODE HERE ###
        
        # Use the ranges above and self.num_tiles to set position_scale and velocity_scale
        # position_scale = number of tiles / position range
        # velocity_scale = number of tiles / velocity range
        
        # Scale position and velocity by multiplying the inputs of each by their scale
        
        ### START CODE HERE ###
        position_scale = self.num_tiles / (position_max - position_min)
        velocity_scale = self.num_tiles / (velocity_max - velocity_min)
        ### END CODE HERE ###
        
        # get the tiles using tc.tiles, with self.iht, self.num_tilings and [scaled position, scaled velocity]
        # nothing to implment here
        tiles = tiles(self.iht, self.num_tilings, [position * position_scale, 
                                                      velocity * velocity_scale])
        
        return np.array(tiles)

