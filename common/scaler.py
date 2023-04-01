import numpy as np

def scale_array(arr):
    positive_sum = 0
    negative_sum = 0
    
    for elem in arr:
        if elem > 0:
            positive_sum += elem
        elif elem < 0:
            negative_sum += elem
            
    if negative_sum >= -1:
        scaling_factor = 1 / positive_sum
    else:
        positive_scaling_factor = 1 / positive_sum
        negative_scaling_factor = -1 / negative_sum
        scaling_factor = np.array([positive_scaling_factor if elem > 0 else negative_scaling_factor for elem in arr])
    
    scaled_arr = np.multiply(arr, scaling_factor)
    
    return scaled_arr


res = scale_array([0.5, 0.7, 0.6, -0.9, -0.3])
print(res, sum(res))