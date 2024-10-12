import numpy as np
from collections import defaultdict
import zlib, math

def compress_data(δt, num_bits = 10, threshhold=True, is_float16=False):
    """
    @param δt : The delta to compress.
    @param num_bits : The number of bits to limit huffman encoded variables to.
    @param treshold : Enabler for priority promotion process.
    @param is_float16 : Enabler for float16 compression.

    @return Zlib compressed promoted delta and uncompressed version.
    """
    _, δt_exp = np.frexp(δt)
    δt_sign = np.sign(δt)
    δt_sign[δt_sign > 0] = 0
    δt_sign[δt_sign < 0] = 1    
    mp =  defaultdict(list)
    for i in range(len(δt)):
        mp[(δt_exp[i], δt_sign[i])].append((i, δt[i]))
    
    # Exponent-base Quantization
    # It partitions entries in δ into multiple buckets according to exponent e and sign s.
    # I.e., it assigns the elements with identical exponents and signs to the same bucket
    for k in mp:
        # represents each bucket by the average of maximum and minimum values in the bucket
        mp[k] = (np.average(np.array([x[-1] for x in mp[k]])), 
                 [x[0] for x in mp[k]])
    mp = list(mp.values())
    
    # Priority Promotion
    if threshhold:
        allowed_buckets = int(math.pow(2, num_bits) - 1) # number of buckets for priority promotion process
        mp = sorted(mp, key = lambda x : abs(x[0]), reverse = True)[:min(allowed_buckets, len(mp))] # sort the results for huffman coding
    new_δt= [0 for x in range(len(δt))]
    for qtVal, pos in mp:
        for p in pos:
            new_δt[p] = qtVal
    
    if is_float16:
        new_δt = np.array(new_δt, dtype = np.float16)
    else:
        new_δt = np.array(new_δt, dtype = np.float32)

    # Huffman Coding
    return zlib.compress(new_δt), new_δt