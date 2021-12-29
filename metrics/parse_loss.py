from loss import objectives

def get_val_at_idx(line, idx):
    return line.split()[idx].strip(',')

if __name__ == '__main__':
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
