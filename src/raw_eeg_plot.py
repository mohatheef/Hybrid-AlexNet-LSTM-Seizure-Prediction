import os
import mne
import matplotlib.pyplot as plt

def plot_eeg_from_edf(edf_file, start=0, duration=10, save_fig=False, fig_dir="figures"):
    """
    Reads and visualizes EEG data from an EDF file.

    Parameters:
        edf_file (str): Path to the EDF file.
        start (int): Start time in seconds for visualization.
        duration (int): Duration in seconds to display.
        save_fig (bool): If True, saves the figure as an image.
        fig_dir (str): Directory to save figures.
    """
    print(f"Processing: {edf_file}")
    raw = mne.io.read_raw_edf(edf_file, preload=True)
    raw.plot(start=start, duration=duration, scalings='auto', title=os.path.basename(edf_file))
    
    if save_fig:
        os.makedirs(fig_dir, exist_ok=True)
        fig_path = os.path.join(fig_dir, os.path.basename(edf_file).replace('.edf', '.png'))
        plt.savefig(fig_path)
        print(f"Saved figure to: {fig_path}")
    
    plt.show()

def process_all_edf_files(root_folder, start=0, duration=10, save_fig=False):
    """
    Walks through the root_folder and processes all .edf files found.
    
    Parameters:
        root_folder (str): Root directory containing patient subfolders.
    """
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith('.edf'):
                edf_path = os.path.join(subdir, file)
                plot_eeg_from_edf(edf_path, start=start, duration=duration, save_fig=save_fig)

# Example usage
root_data_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "seizure_data")
process_all_edf_files(root_data_folder, start=0, duration=10, save_fig=False)
