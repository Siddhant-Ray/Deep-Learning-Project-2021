objectives = [{
                'name': 'ResNet50',
                'input_file': '../resnet/resnet_results/scratch_resnet50.txt',
                'output_folder': 'resnet',
                'train_loss_key': 'train loss',
                'train_loss_idx': 2,
                'train_acc_idx': 4,
                'validation_loss_key': 'validation loss',
                'validation_loss_idx': 2,
                'validation_acc_idx': 4
              },
              {
                'name': 'ViT',
                'input_file': '../vit/vit_2021_12_09_run_scratch_mixed_prec.txt',
                'output_folder': 'vit',
                'train_loss_key': 'train_loss',
                'train_loss_idx': 3,
                'train_acc_idx': 5,
                'validation_loss_key': 'eval_loss',
                'validation_loss_idx': 3,
                'validation_acc_idx': 5
              }]