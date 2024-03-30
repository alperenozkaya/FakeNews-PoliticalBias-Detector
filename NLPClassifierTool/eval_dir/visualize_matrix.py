import numpy as np
import matplotlib.pyplot as plt


# Function to read the confusion matrix from a file
def read_confusion_matrix(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Parsing the header for labels
    labels = lines[0].split()

    # Initializing an empty matrix
    matrix = []
    for line in lines[1:]:  # Skipping the first line which is the header
        row = line.split()[1:]  # Skipping the label part
        matrix.append([int(num) for num in row])

    return np.array(matrix), labels


def main():
    file_path = 'all_confusion_matrix'  # Update this path accordingly

    # Reading the confusion matrix and labels
    confusion_matrix, labels = read_confusion_matrix(file_path)

    # Visualization
    fig, ax = plt.subplots()
    cax = ax.matshow(confusion_matrix, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    plt.xticks(rotation=45)

    for (i, j), val in np.ndenumerate(confusion_matrix):
        ax.text(j, i, f'{val}', ha='center', va='center', color='black')

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix Visualization')
    plt.show()


if __name__ == '__main__':
    main()
