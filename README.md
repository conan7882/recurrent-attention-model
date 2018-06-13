# recurrent-visual-attention-tensorflow
A TensorFlow implementation of the recurrent models of visual attention

## Gradient flow
- Cross entropy loss: train the action network, core network and glimpse network.
- REINFORCE: train the location network (stop gradient: reward, core network state)
- MSE loss of baseline: train paramters of baseline estimation