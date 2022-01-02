import numpy as np
import matplotlib.pyplot as plt

from loss import objectives


TARGETS = ['train_loss', 'train_acc', 'validation_loss', 'validation_acc']
TARGET_YLABELS = ['Loss', 'Accuracy', 'Loss', 'Accuracy']
TARGET_TITLES = ['Training loss', 'Training accuracy', 'Test loss', 'Test accuracy']
NO_OF_EPOCHS = 80
COLOURS = ['red', 'green', 'blue']


def load_curve(input_file):
    curve = np.loadtxt(input_file)
    return curve


def plot_loss_curves(curves, labels, colors, output_path, name='loss_curve', ylabel='Loss', title='Loss'):
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    
    for curve, label, color in zip(curves, labels, colors):
        ax.plot(range(len(curve)), curve, color=color, label=label)

    ax.tick_params(axis='both', which='major', labelsize='x-large')
    ax.set_xlim(0, len(curve)-1)
    ax.set_xlabel('Epoch', fontsize='x-large')
    ax.set_ylabel(ylabel, fontsize='x-large')
    ax.legend(fontsize='x-large')
    ax.set_title(title, fontsize='x-large')

    plt.savefig(output_path+'/'+name+'.pdf', dpi=1000, bbox_inches='tight')


if __name__ == '__main__':    
    for objective in objectives:
        for target, target_ylabel, target_title in zip(TARGETS, TARGET_YLABELS, TARGET_TITLES):
            curve = load_curve(objective['output_folder']+'/'+target+'.csv')[:NO_OF_EPOCHS]
            plot_loss_curves([curve], [objective['name']], ['red'], \
                objective['output_folder'], name=target, ylabel=target_ylabel, title=target_title)
    
    for target, target_ylabel, target_title in zip(TARGETS, TARGET_YLABELS, TARGET_TITLES):
        curves = [load_curve(objective['output_folder']+'/'+target+'.csv')[:NO_OF_EPOCHS] for objective in objectives]
        names = [objective['name'] for objective in objectives]
        plot_loss_curves(curves, names, COLOURS[:len(curves)], \
            '.', name=target+'_combined', ylabel=target_ylabel, title=target_title)