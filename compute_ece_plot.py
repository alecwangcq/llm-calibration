import argparse
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
from sklearn.utils import resample
from tqdm import tqdm

# def compute_ece(bins, confidences, accuracies, num_samples):
#     ece = 0.0
#     for bin_lower, bin_upper in bins:
#         # Find indices of samples that fall into the current bin
#         indices = np.where((confidences > bin_lower) & (confidences <= bin_upper))[0]
#         if len(indices) == 0:
#             continue
#         # Calculate average confidence and accuracy within the bin
#         avg_confidence = np.mean(confidences[indices])
#         avg_accuracy = np.mean(accuracies[indices])
#         # Weight by the proportion of samples in the bin
#         weight = len(indices) / num_samples
#         # Accumulate the ECE
#         ece += np.abs(avg_confidence - avg_accuracy) * weight
#     return ece

def compute_ece(bins, confidences, accuracies, num_samples):
    ece = 0.0
    for bin_lower, bin_upper in bins:
        # Find indices of predictions that fall into the current bin
        indices = np.where((confidences >= bin_lower) & (confidences < bin_upper))[0]
        if len(indices) == 0:
            continue
        # Calculate average confidence and accuracy within the bin
        avg_confidence = np.mean(confidences[indices])
        avg_accuracy = np.mean(accuracies[indices])
        # Weight by the proportion of samples in the bin
        weight = len(indices) / num_samples
        # Accumulate the ECE
        ece += np.abs(avg_confidence - avg_accuracy) * weight
    return ece

def bootstrap_accuracy(accuracies, n_bootstrap=1000):
    # Perform bootstrap resampling to compute confidence intervals
    mean_accuracies = []
    for _ in range(n_bootstrap):
        sample = resample(accuracies, replace=True, n_samples=len(accuracies))
        mean_acc = np.mean(sample)
        mean_accuracies.append(mean_acc)
    # Compute 2.5th and 97.5th percentiles for 95% CI
    lower = np.percentile(mean_accuracies, 2.5)
    upper = np.percentile(mean_accuracies, 97.5)
    return lower, upper

def plot_reliability_diagram(bins, confidences, accuracies, output_path, ece):
    bin_centers = []
    accuracy_per_bin = []
    confidence_per_bin = []
    lower_bounds = []
    upper_bounds = []
    for bin_lower, bin_upper in bins:
        indices = np.where((confidences > bin_lower) & (confidences <= bin_upper))[0]
        bin_center = (bin_lower + bin_upper) / 2
        bin_centers.append(bin_center)

        if len(indices) == 0:
            # If no samples in the bin, append NaNs
            accuracy_per_bin.append(np.nan)
            confidence_per_bin.append(np.nan)
            lower_bounds.append(np.nan)
            upper_bounds.append(np.nan)
            continue

        avg_confidence = np.mean(confidences[indices])
        avg_accuracy = np.mean(accuracies[indices])
        # Bootstrap to compute 95% CI for accuracy
        accs_in_bin = accuracies[indices]
        lower, upper = bootstrap_accuracy(accs_in_bin)

        confidence_per_bin.append(avg_confidence)
        accuracy_per_bin.append(avg_accuracy)
        lower_bounds.append(lower)
        upper_bounds.append(upper)

    # Convert lists to numpy arrays for plotting
    bin_centers = np.array(bin_centers)
    accuracy_per_bin = np.array(accuracy_per_bin)
    lower_bounds = np.array(lower_bounds)
    upper_bounds = np.array(upper_bounds)

    # Plotting the bar plot with error bars
    plt.figure(figsize=(10, 10))
    bar_width = 0.08  # Width of the bars
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.bar(bin_centers, accuracy_per_bin, width=bar_width, color='skyblue', edgecolor='black', label='Observed Accuracy')

    # Plot error bars for 95% CI
    plt.errorbar(bin_centers, accuracy_per_bin, yerr=[accuracy_per_bin - lower_bounds, upper_bounds - accuracy_per_bin],
                 fmt='none', ecolor='black', capsize=5, label='95% Confidence Interval')

    # Plot the perfect calibration line
    plt.plot([0, 1], [0, 1], linestyle='--', color='red', label='Perfect Calibration')
    plt.text(0.05, 0.95, f'ECE = {ece:.4f}', fontsize=14, fontweight='bold', verticalalignment='top')

    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Model Calibration Bar Plot with 95% Confidence Intervals')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path.replace('.csv', '_ece_bar_plot.png'))
    plt.close()

# def process_csv_files(csv_files):
#     all_confidences = []
#     all_accuracies = []
#     num_samples = 0
#     for csv_file in tqdm(csv_files, desc='Processing CSV files'):
#         df = pd.read_csv(csv_file)

#         # Extract logits and compute probabilities
#         logits = df[['Logit_A', 'Logit_B', 'Logit_C', 'Logit_D']].values
#         probabilities = softmax(logits, axis=1)  # Compute softmax probabilities

#         # Get predicted probabilities and classes
#         predicted_probs = probabilities.max(axis=1)
#         predicted_classes = probabilities.argmax(axis=1)

#         # Map correct answers to indices
#         answer_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
#         correct_answers = df.iloc[:, 5].map(lambda x: answer_map.get(str(x).strip(), -1)).values

#         # Determine correctness of predictions
#         correct_predictions = (predicted_classes == correct_answers).astype(float)

#         # Accumulate confidences and accuracies
#         all_confidences.extend(predicted_probs)
#         all_accuracies.extend(correct_predictions)
#         num_samples += len(df)

#     all_confidences = np.array(all_confidences)
#     all_accuracies = np.array(all_accuracies)
#     return all_confidences, all_accuracies, num_samples

def process_csv_files(csv_files):
    all_confidences = []
    all_accuracies = []
    num_classes = 4  # Assuming 4 classes: A, B, C, D
    for csv_file in tqdm(csv_files, desc='Processing CSV files'):
        df = pd.read_csv(csv_file)

        # Extract logits and compute probabilities
        logits = df[['Logit_A', 'Logit_B', 'Logit_C', 'Logit_D']].values
        probabilities = softmax(logits, axis=1)  # Compute softmax probabilities

        # Map correct answers to indices
        answer_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        correct_answers = df.iloc[:, 5].map(lambda x: answer_map.get(str(x).strip(), -1)).values

        # One-hot encode the correct answers
        correct_one_hot = np.zeros_like(probabilities)
        correct_one_hot[np.arange(len(correct_answers)), correct_answers] = 1

        # Accumulate confidences and accuracies for all classes
        all_confidences.extend(probabilities.flatten())
        all_accuracies.extend(correct_one_hot.flatten())

    all_confidences = np.array(all_confidences)
    all_accuracies = np.array(all_accuracies)
    num_samples = len(all_confidences)

    return all_confidences, all_accuracies, num_samples

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_dir', type=str, required=True, help='Directory containing the CSV files.')
    args = parser.parse_args()

    # Define bins for confidence intervals [0, 0.1], (0.1, 0.2], ..., (0.9, 1.0]
    bins = [(i/10.0, (i+1)/10.0) for i in range(10)]

    # Gather all logits CSV files
    csv_files = glob.glob(f'{args.csv_dir}/*_logits.csv')

    for f in csv_files:
        # Process CSV files to get confidences and accuracies
        confidences, accuracies, total_samples = process_csv_files([f])
        # confidences, accuracies, total_samples = process_csv_files(csv_files)

        # Compute Expected Calibration Error (ECE)
        ece = compute_ece(bins, confidences, accuracies, total_samples)
        print(f"\nExpected Calibration Error (ECE): {ece:.4f}")

        # Plot and save the calibration bar plot with 95% CI
        # sample_csv = csv_files[0] if csv_files else 'ece_plot.png'
        sample_csv = f
        plot_reliability_diagram(bins, confidences, accuracies, sample_csv, ece)

        print(f"\nCalibration bar plot saved as {sample_csv.replace('.csv', '_ece_bar_plot.png')}")
    
    # Process CSV files to get confidences and accuracies
    if True:
        confidences, accuracies, total_samples = process_csv_files(csv_files)
        # confidences, accuracies, total_samples = process_csv_files(csv_files)

        # Compute Expected Calibration Error (ECE)
        ece = compute_ece(bins, confidences, accuracies, total_samples)
        print(f"\nExpected Calibration Error (ECE): {ece:.4f}")

        # Plot and save the calibration bar plot with 95% CI
        # sample_csv = csv_files[0] if csv_files else 'ece_plot.png'
        sample_csv = os.path.join(args.csv_dir, 'overall.csv')
        plot_reliability_diagram(bins, confidences, accuracies, sample_csv, ece)

        print(f"\nCalibration bar plot saved as {sample_csv.replace('.csv', '_ece_bar_plot.png')}")

# import argparse
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import glob
# from scipy.special import softmax

# def compute_ece(bins, confidences, accuracies, num_samples):
#     ece = 0.0
#     for bin_lower, bin_upper in bins:
#         # Find indices of samples that fall into the current bin
#         indices = np.where((confidences > bin_lower) & (confidences <= bin_upper))[0]
#         if len(indices) == 0:
#             continue
#         # Calculate average confidence and accuracy within the bin
#         avg_confidence = np.mean(confidences[indices])
#         avg_accuracy = np.mean(accuracies[indices])
#         # Weight by the proportion of samples in the bin
#         weight = len(indices) / num_samples
#         # Accumulate the ECE
#         ece += np.abs(avg_confidence - avg_accuracy) * weight
#     return ece

# def plot_reliability_diagram(bins, confidences, accuracies, output_path):
#     bin_centers = []
#     accuracy_per_bin = []
#     confidence_per_bin = []
#     for bin_lower, bin_upper in bins:
#         indices = np.where((confidences > bin_lower) & (confidences <= bin_upper))[0]
#         if len(indices) == 0:
#             continue
#         avg_confidence = np.mean(confidences[indices])
#         avg_accuracy = np.mean(accuracies[indices])
#         bin_centers.append((bin_lower + bin_upper) / 2)
#         confidence_per_bin.append(avg_confidence)
#         accuracy_per_bin.append(avg_accuracy)
#     plt.figure(figsize=(8, 8))
#     plt.xlim(0, 1)
#     plt.ylim(0, 1)
#     plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration')
#     plt.plot(confidence_per_bin, accuracy_per_bin, marker='o', label='Model Calibration')
#     plt.xlabel('Confidence')
#     plt.ylabel('Accuracy')
#     plt.title('Reliability Diagram')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(output_path.replace('.csv', '_ece_plot.png'))
#     plt.close()

# def process_csv_files(csv_files):
#     all_confidences = []
#     all_accuracies = []
#     num_samples = 0
#     for csv_file in csv_files:
#         print(f"Processing {csv_file}.")
#         df = pd.read_csv(csv_file)
#         logits = df[['Logit_A', 'Logit_B', 'Logit_C', 'Logit_D']].values
#         # Compute softmax probabilities with numerical stability
#         probabilities = softmax(logits, axis=1)
#         # Get the predicted probabilities and predicted classes
#         predicted_probs = probabilities.max(axis=1)
#         predicted_classes = probabilities.argmax(axis=1)
#         # Map correct answers to indices
#         answer_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
#         correct_answers = df.iloc[:, 5].map(lambda x: answer_map.get(x.strip(), -1)).values
#         # Determine correctness of predictions
#         correct_predictions = (predicted_classes == correct_answers).astype(float)
#         # Accumulate results
#         all_confidences.extend(predicted_probs)
#         all_accuracies.extend(correct_predictions)
#         num_samples += len(df)
#     all_confidences = np.array(all_confidences)
#     all_accuracies = np.array(all_accuracies)
#     return all_confidences, all_accuracies, num_samples

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--csv_dir', type=str, required=True, help='Directory containing the CSV files.')
#     args = parser.parse_args()

#     # Define bins for confidence intervals
#     bins = [(i / 10.0, (i + 1) / 10.0) for i in range(10)]
#     csv_files = glob.glob(f'{args.csv_dir}/*_logits.csv')

#     confidences, accuracies, total_samples = process_csv_files(csv_files)
#     ece = compute_ece(bins, confidences, accuracies, total_samples)
#     print(f"\nExpected Calibration Error (ECE): {ece:.4f}")

#     # Plot and save the reliability diagram
#     sample_csv = csv_files[0] if csv_files else 'ece_plot.png'
#     plot_reliability_diagram(bins, confidences, accuracies, sample_csv)

#     print(f"ECE plot saved as {sample_csv.replace('.csv', '_ece_plot.png')}")