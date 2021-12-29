def get_val_at_idx(line, idx):
    return line.split()[idx].strip(',')

objectives = [{
                'name': 'resnet',
                'input_file': 'resnet/resnet_results/scratch_resnet50.txt',
                'output_folder': 'metrics/resnet',
                'train_loss_key': 'train loss',
                'train_loss_idx': 2,
                'train_acc_idx': 4,
                'validation_loss_key': 'validation loss',
                'validation_loss_idx': 2,
                'validation_acc_idx': 4
              },
              {
                'name': 'vit',
                'input_file': 'vit/vit_2021_12_09_run_scratch_mixed_prec.txt',
                'output_folder': 'metrics/vit',
                'train_loss_key': 'train_loss',
                'train_loss_idx': 3,
                'train_acc_idx': 5,
                'validation_loss_key': 'eval_loss',
                'validation_loss_idx': 3,
                'validation_acc_idx': 5
              },
              
              ]

for objective in objectives:
    with open(objective['input_file'], 'r') as input_file, \
        open(objective['output_folder'] + '/train_loss.csv', 'w') as train_loss_file, \
        open(objective['output_folder'] + '/train_acc.csv', 'w') as train_acc_file, \
        open(objective['output_folder'] + '/validation_loss.csv', 'w') as validation_loss_file, \
        open(objective['output_folder'] + '/validation_acc.csv', 'w') as validation_acc_file:
        for line in input_file:
            if objective['train_loss_key'] in line:
                train_loss = get_val_at_idx(line, objective['train_loss_idx'])
                train_loss_file.write(str(float(train_loss)) + '\n')
                train_acc = get_val_at_idx(line, objective['train_acc_idx'])
                train_acc_file.write(str(float(train_acc)) + '\n')
            if objective['validation_loss_key'] in line:
                validation_loss = get_val_at_idx(line, objective['validation_loss_idx'])
                validation_loss_file.write(str(float(validation_loss)) + '\n')
                validation_acc = get_val_at_idx(line, objective['validation_acc_idx'])
                validation_acc_file.write(str(float(validation_acc)) + '\n')
