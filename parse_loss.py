import numpy as np

## Resnet50

RESNET50_INPUT = 'resnet/resnet_results/scratch_resnet50.txt'
RESNET50_OUTPUT = 'metrics/resnet'

with open(RESNET50_INPUT, 'r') as resnet50_input, \
    open(RESNET50_OUTPUT + '/train_loss.csv', 'w') as train_loss_file, \
    open(RESNET50_OUTPUT + '/train_acc.csv', 'w') as train_acc_file, \
    open(RESNET50_OUTPUT + '/validation_loss.csv', 'w') as validation_loss_file, \
    open(RESNET50_OUTPUT + '/validation_acc.csv', 'w') as validation_acc_file:
    for line in resnet50_input:
        if 'train loss' in line:
            train_loss = line[12:18]
            train_loss_file.write(str(float(train_loss)) + '\n')
            train_acc = line[25:31]
            train_acc_file.write(str(float(train_acc)) + '\n')
        if 'validation loss' in line:
            validation_loss = line[17:23]
            validation_loss_file.write(str(float(validation_loss)) + '\n')
            validation_acc = line[30:36]
            validation_acc_file.write(str(float(validation_acc)) + '\n')
