
import numpy as np
from scipy import stats

TRUTH_LABELS_FILE = "./datasets/combined_cifar_eval/labels.csv"
RESNET_SOFTMAX_FILE = "./resnet/resnet_results/softmax_probs.csv"
VIT_SOFTMAX_FILE = "./vit/classification_2021-12-22_12:27:11/softmax_probs.csv"
METRICS_DIR = "./metrics/"


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
	truth_max = np.argsort(truth, axis=1)[:, :4]
	outputs_max = np.argsort(outputs, axis=1)[:, :4]

	top_4_matching = np.empty(truth.shape[0])
	for i, (t, o) in enumerate(zip(truth_max, outputs_max)):
		top_4_matching[i] = len(np.intersect1d(t, o))

	average_matching = np.average(top_4_matching)
	return average_matching, top_4_matching

def output(name, truth, predictions, save_files = True):
	euc_avg, euc = euclidian_metrics(truth, predictions)
	ws_avg, ws = wasserstein_metrics(truth, predictions)
	t4_avg, t4 = top_4_metrics(truth, predictions)

	print("average euclidian norm: ", euc_avg)
	print("average wasserstein norm: ", ws_avg)
	print("average top 4 matching: ", t4_avg)

	if save_files:
		np.savetxt(METRICS_DIR + name + "_euclidian.csv", euc)
		np.savetxt(METRICS_DIR + name + "_wasserstein.csv", ws)
		np.savetxt(METRICS_DIR + name + "_top4.csv", t4)

def main():
	truth_labels = np.loadtxt(TRUTH_LABELS_FILE, dtype=float)
	resnet_softmax = np.loadtxt(RESNET_SOFTMAX_FILE, dtype=float)
	vit_softmax = np.loadtxt(VIT_SOFTMAX_FILE, dtype=float)

	noise_labels = np.copy(truth_labels)
	np.random.shuffle(noise_labels)

	print("SUMMARY")
	print("- RESNET COMPARED TO TRUE LABELS ---------------")
	output("resnet", truth_labels, resnet_softmax)

	print("- RESNET COMPARED TO RANDOM NOISE ------------------")
	output("resnet_noise", noise_labels, resnet_softmax, save_files=False)

	print("- VIT COMPARED TO TRUE LABELS ------------------")
	output("vit", truth_labels, vit_softmax)

	print("- VIT COMPARED TO RANDOM NOISE ------------------")
	output("vit_noise", noise_labels, vit_softmax, save_files=False)

if __name__ == "__main__":
	main()
