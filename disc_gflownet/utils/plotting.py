import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import os


def plot_loss_curve(losses_A, losses_B=None, zs=None, title="", save_dir=None):
    if zs is not None:
        fig, ax = plt.subplots(2, 1, figsize=(10, 6))
        plt.sca(ax[0])
    else:
        plt.figure(figsize=(10,3))

    if isinstance(losses_B, type(None)):
        plt.plot(losses_A, color="black")
    else:
        plt.plot(losses_A, color="blue", linewidth=1, label="No Forward Masks")
        plt.plot(losses_B, color="red", linewidth=1, label="Forward Masks", alpha=0.5)
        plt.legend()

    plt.yscale('log')
    plt.ylabel('Loss')
    
    if zs is not None:
        plt.sca(ax[1])
        plt.plot(zs, color="black")
        plt.ylabel('Estimated Z')
        plt.xlabel('Training Steps')
        plt.suptitle(title)
    else:
        plt.xlabel('Training Steps')
        plt.title(title)

    if save_dir is not None:
        output_img_loss = os.path.join(save_dir, "loss_curve.png")
        plt.savefig(output_img_loss)
        plt.close()
    else:
        plt.show()

