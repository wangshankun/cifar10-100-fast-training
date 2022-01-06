import numpy as np
from training import *
import apex.amp as amp
import resnet

import warnings
warnings.filterwarnings("ignore")

epochs = 70
N_runs = 2

batch_size = 512

#train_batches, test_batches= get_cifar10_date_batches(batch_size_train = batch_size, batch_size_test = batch_size)

train_batches, test_batches= get_cifar100_date_batches(batch_size=512)

summaries = []
for i in range(N_runs):
    model = resnet.resnet18(num_classes=100).cuda()
    summaries.append(training(model,epochs = epochs, batches = (train_batches, test_batches), batch_size = batch_size,  opt_level='O1'))

test_accs = np.array([s['test acc'] for s in summaries])
print(f'mean test accuracy: {np.mean(test_accs):.4f}')
print(f'median test accuracy: {np.median(test_accs):.4f}')

