
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


TRUTH_LABELS_FILE = "../datasets/combined_cifar_eval/labels.csv"
RESNET_SOFTMAX_FILE = "../resnet/resnet_results/softmax_probs.csv"
VIT_SOFTMAX_FILE = "../vit/classification_combined_2021-12-22_12:27:11/softmax_probs.csv"
LOCAL_VIT_SOFTMAX_FILE = "../DemystifyLocalViT/DemystifyLocal_results/softmax_probs.csv"
METRICS_DIR = "./combined_image_performance/"

np.random.seed(1234)

def euclidian_metrics(truth, outputs):
    diff = truth - outputs
    norms = np.linalg.norm(diff, axis=1)

    average_norm = np.average(norms)
    return average_norm, norms


def wasserstein_metrics(truth, outputs):
    norms = np.empty(truth.shape[0])
    for i, (t, o) in enumerate(zip(truth, outputs)):
        norms[i] = stats.wasserstein_distance(t, o)

    average_norm = np.average(norms)
    return average_norm, norms

def top_4_metrics(truth, outputs):
    truth_max = np.argsort(truth, axis=1)[:, -4:]
    outputs_max = np.argsort(outputs, axis=1)[:, -4:]

    top_4_matching = np.empty(truth.shape[0])
    for i, (t, o) in enumerate(zip(truth_max, outputs_max)):
        top_4_matching[i] = len(np.intersect1d(t, o))

    average_matching = np.average(top_4_matching)
    return average_matching, top_4_matching

def output(name, displayname, truth, predictions, save_files = True):
    euc_avg, euc = euclidian_metrics(truth, predictions)
    # ws_avg, ws = wasserstein_metrics(truth, predictions)
    t4_avg, t4 = top_4_metrics(truth, predictions)
    t4_histogram, bins = np.histogram(t4, bins=[0, 1, 2, 3, 4, 5])

    print("average euclidian norm: ", euc_avg)
    # print("average wasserstein norm: ", ws_avg)
    print("average top 4 matching: ", t4_avg)
    print("top 4 matching histogram: ", t4_histogram)

    if save_files:
        np.savetxt(METRICS_DIR + name + "_euclidian.csv", euc)
        # np.savetxt(METRICS_DIR + name + "_wasserstein.csv", ws)
        np.savetxt(METRICS_DIR + name + "_top4.csv", t4)
        histo_x, histo_y, bars = plt.hist(t4, [b - .5 for b in bins], edgecolor='black')

        for l, rect in zip(t4_histogram, bars):
            pct = (l / sum(t4_histogram))
            plt.text(rect.get_x() + rect.get_width()/2, rect.get_height()+0.01, f"{pct:.2%}", ha='center', va='bottom')

        plt.title(displayname + " - Top 4 Matching Histogram")
        plt.savefig(METRICS_DIR + name + "_top4_histogram.pdf", transparent=False)
        plt.clf()

def main():
    truth_labels = np.loadtxt(TRUTH_LABELS_FILE, dtype=float)
    resnet_softmax = np.loadtxt(RESNET_SOFTMAX_FILE, dtype=float)
    vit_softmax = np.loadtxt(VIT_SOFTMAX_FILE, dtype=float)
    local_vit_softmax = np.loadtxt(LOCAL_VIT_SOFTMAX_FILE, dtype=float)

    noise_labels = np.copy(truth_labels)
    for row in noise_labels:
        np.random.shuffle(row)

    print("SUMMARY")
    print("- RESNET COMPARED TO TRUE LABELS ---------------")
    output("resnet", "ResNet", truth_labels, resnet_softmax)

    print("- RESNET COMPARED TO RANDOM NOISE ------------------")
    output("resnet_noise", "ResNet (Noise)", noise_labels, resnet_softmax, save_files=False)

    print("- VIT COMPARED TO TRUE LABELS ------------------")
    output("vit", "Vision Transformer", truth_labels, vit_softmax)

    print("- VIT COMPARED TO RANDOM NOISE ------------------")
    output("vit_noise", "Vision Transformer (Noise)", noise_labels, vit_softmax, save_files=False)

    print("- LOCAL VIT COMPARED TO TRUE LABELS ------------------")
    output("local_vit", "Local Vision Transformer", truth_labels, local_vit_softmax)

    print("- LOCAL VIT COMPARED TO RANDOM NOISE ------------------")
    output("local_vit_noise", "Local Vision Transformer (Noise)", noise_labels, local_vit_softmax, save_files=False)

if __name__ == "__main__":
    main()
