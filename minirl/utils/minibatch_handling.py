import numpy as np


def generate_minibatch(batch_size, *dataset, num_targets=1):
    size = len(dataset[0])
    num_batches = int(np.ceil(size/batch_size))
    batches_intervals = [(i * batch_size, min(size, (i+1)*batch_size))
                         for i in range(num_batches)]
    
    indixes = np.arange(0, size, 1)
    np.random.shuffle(indixes)

    for start, end in batches_intervals:
        data = []
        for i in range(len(dataset)):
            data.append(dataset[i][indixes[start:end]])

        yield data