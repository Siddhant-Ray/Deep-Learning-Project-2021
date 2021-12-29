import numpy as np
import matplotlib.pyplot as plt

from loss import objectives


TARGETS = ['train_loss', 'train_acc', 'validation_loss', 'validation_acc']
NO_OF_EPOCHS = 80


def load_curve(input_file):
    curve = np.loadtxt(input_file)
    return curve


def plot_loss_curves(curves, labels, colors, no_of_epochs, output_path, name='loss_curve'):
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    
    for curve, label, color in zip(curves, labels, colors):
        ax.plot(range(no_of_epochs), curve, color=color, label=label)

    ax.set_xlim(0, no_of_epochs-1)
    ax.set_xlabel('Epoch', fontsize='large')
    ax.set_ylabel('Loss', fontsize='large')
    ax.legend(fontsize='large')

    plt.savefig(output_path+'/'+name+'.pdf', dpi=1000, bbox_inches='tight')


if __name__ == '__main__':
    for objective in objectives:
        for target in TARGETS:
            curve = load_curve(objective['output_folder']+'/'+target+'.csv')[:NO_OF_EPOCHS]
            plot_loss_curves([curve], [objective['name']], ['red'], NO_OF_EPOCHS, \
                objective['output_folder'], name=target)