# import libraries
import re
import matplotlib.pyplot as plt
import ast
import pandas as pd

def extract_losses_epoch_end(line):
    match = re.search(r'(\{.*\})', line)
    if match:
        epoch_info_str = match.group(1)
        epoch_info = ast.literal_eval(epoch_info_str)
        return epoch_info.get('train_loss'), epoch_info.get('val_loss')
    
    else:
        return None

def plot_losses(train_losses, val_losses, emb_type):
    # create figure
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f"[{emb_type}] Train and Validation Loss Over Epochs")
    plt.legend()

    # set ylim
    plt.ylim(0.4, 1.2)
    
    # save figure
    filename = f"./figures/{emb_type}_losses.png"
    plt.savefig(filename)
    plt.close()

def plot_all_losses(train_losses, val_losses, emb_types):
    # create figure
    fig, axs = plt.subplots(2, 2)
    for ax, train_loss, val_loss, emb_type in zip(axs.flatten(), train_losses, val_losses, emb_types):
        ax.plot(train_loss, label="Train Loss")
        ax.plot(val_loss, label="Validation Loss")
    
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(emb_type)
        ax.set_ylim(0.4, 1.2)
        ax.legend()
    
    # save figure
    fig.tight_layout()
    #fig.suptitle("Train and Validation Loss")
    filename = f"./figures/all_losses.png"
    fig.savefig(filename, dpi=1000)

def save_csv(train_losses, val_losses, emb_type):
    # convert loss values to dictionary
    losses_dict = {"train": train_losses, "val": val_losses}
    losses_df = pd.DataFrame(losses_dict)

    # define filename
    filename = f"./data/{emb_type}_losses.csv"
    losses_df.to_csv(filename)

def main(out_file, emb_type):

    # open file and read
    with open(out_file, 'r') as file:
        lines = file.readlines()

    # loop through all lines and store loss values
    train_losses = []
    val_losses = []
    for line in lines:
        losses = extract_losses_epoch_end(line)
        if losses is not None:
            train_loss, val_loss = losses
            if train_loss < 1.5 and val_loss < 1.5:
                train_losses.append(train_loss)
                val_losses.append(val_loss)

    # save csv files
    plot_losses(train_losses = train_losses, val_losses = val_losses, emb_type=emb_type)
    save_csv(train_losses=train_losses, val_losses=val_losses, emb_type = emb_type)
    
    return train_losses, val_losses, emb_type

if __name__ == "__main__":

    # define out files
    OHE_OUT = "run_train_eigenfold_one_hot_new.out"
    OMEGA_OUT = "run_train_eigenfold.out"
    PROTTRANS_OUT = "run_train_eigenfold_prottrans.out"
    ESM_OUT = "run_train_eigenfold_esm.out"

    # out files dictionary
    out_files = {"One-Hot": OHE_OUT, "Omegafold": OMEGA_OUT, "ProtT5": PROTTRANS_OUT, "ESM": ESM_OUT}

    TYPE = "all" # change for type of embedding

    if TYPE != "all":
        # define arguments for filepath and csv/loss files
        out_file = f"../out/{out_files[TYPE]}" 
        main(out_file=out_file, emb_type = TYPE)

    else:
        # store combined losses
        all_train_losses = []
        all_val_losses = []
        all_emb_type = []

        for key in out_files:
            out_file = f"../out/{out_files[key]}" 
            train_losses, val_losses, emb_type = main(out_file=out_file, emb_type = key)

            # apend vals
            all_train_losses.append(train_losses)
            all_val_losses.append(val_losses)
            all_emb_type.append(emb_type)

        # plot all figs
        plot_all_losses(all_train_losses, all_val_losses, all_emb_type)
