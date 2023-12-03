import re
import matplotlib.pyplot as plt

def extract_loss(line):
    match = re.search(r'loss (\d+\.\d+)', line)
    if match:
        return float(match.group(1))
    else:
        return None

def plot_losses(loss_values):
    plt.plot(loss_values)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss Over Training Iterations')
    plt.savefig("./figures/omegafold_iteration_loss.png")

def main(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    loss_values = []
    for line in lines:
        loss = extract_loss(line)
        if loss is not None and loss < 2:
            loss_values.append(loss)

    print(loss_values[:5])
    plot_losses(loss_values)

if __name__ == "__main__":
    file_path = "../out/run_train_eigenfold.out"  # Replace with the actual path to your log file
    main(file_path)
