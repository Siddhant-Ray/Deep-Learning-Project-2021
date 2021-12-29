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
                'train_loss_key': '782] train_loss',
                'train_loss_idx': 3,
                'train_acc_idx': 5,
                'validation_loss_key': '157] eval_loss',
                'validation_loss_idx': 3,
                'validation_acc_idx': 5
              },
              {
                'name': 'Local ViT',
                'input_file': '../DemystifyLocalViT/output_for_190_epochs.txt',
                'output_folder': 'local_vit',
                'train_loss_key': '[3120/3125]', # hacky, skipping last 5 images
                'train_loss_idx': 15,
                'validation_loss_key': '[620/625]', # hacky, skipping last 5 images
                'validation_loss_idx': 12,
                'validation_acc_idx': 15
              }]