def normalize(data, mean, stddev, eps=0.):
    return (data - mean) / (stddev + eps)
  
def normalize_instance(data, eps=0.):
    return normalize(data, mean, std, eps), mean, std
