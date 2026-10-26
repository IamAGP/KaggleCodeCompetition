"""Exact-epoch data-parallel batches without duplicating final-window samples."""
import math
import numpy as np


def epoch_windows(size, seed, epoch, rank, world_size, batch_size, accumulation):
    if size < 1 or not 0 <= rank < world_size:
        raise ValueError('Invalid dataset size or rank')
    if min(world_size,batch_size,accumulation)<1:
        raise ValueError('Batch dimensions must be positive')
    order=np.random.default_rng(np.random.SeedSequence([seed,epoch])).permutation(size)
    global_microbatch=world_size*batch_size
    window_size=global_microbatch*accumulation
    for start in range(0,size,window_size):
        window=order[start:start+window_size]
        microsteps=math.ceil(len(window)/global_microbatch)
        steps=[]
        for microstep in range(microsteps):
            offset=microstep*global_microbatch+rank*batch_size
            indices=window[offset:offset+batch_size]
            # DDP averages gradients across ranks. Multiplication by world_size
            # restores the global document-mean gradient, including short tails.
            weight=world_size*len(indices)/len(window)
            steps.append((indices,weight,microstep==microsteps-1))
        yield steps
