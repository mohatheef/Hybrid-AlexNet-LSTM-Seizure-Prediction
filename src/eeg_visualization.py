import os
import mne
import matplotlib.pyplot as plt

# ----------- CONFIGURATION -----------
data_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "seizure_data", "chb01")
summary_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "summaries", "chb01-summary.txt")
fig_dir = "figures"
save_fig = False
pre_ictal_duration = 30   # seconds before seizure start
post_ictal_duration = 30  # seconds after seizure end

# ----------- PARSE SUMMARY FILE -----------
def parse_summary(file_path):
    seizure_info = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()

    file_name = None
    for line in lines:
        line = line.strip()
        if line.startswith("File Name:"):
            file_name = line.split()[-1]
            seizure_info[file_name] = []
        elif line.startswith("Seizure Start Time:"):
            start_time = int(line.split()[-2])
        elif line.startswith("Seizure End Time:"):
            end_time = int(line.split()[-2])
            seizure_info[file_name].append((start_time, end_time))
    return seizure_info

# ----------- PROCESS ONE PATIENT WITH STEP-BY-STEP VISUALIZATION -----------
def process_one_patient(data_root, summary_file, save_fig=False, fig_dir="figures", pre_ictal_duration=30, post_ictal_duration=30):
    seizure_info = parse_summary(summary_file)

    for file in sorted(os.listdir(data_root)):
        if file.lower().endswith(".edf"):
            edf_path = os.path.join(data_root, file)
            print(f"\nLoading: {edf_path}")
            raw = mne.io.read_raw_edf(edf_path, preload=True)
            file_seizures = seizure_info.get(file, [])

            if not file_seizures:
                print(f"No seizure info for: {file}")
                continue

            for i, (start, end) in enumerate(file_seizures):
                print(f"\nSeizure {i+1}: {start}s to {end}s")

                # --- Pre-ictal phase ---
                pre_start = max(0, start - pre_ictal_duration)
                if pre_start < start:
                    print(f"  Pre-ictal {i+1}: {pre_start}s to {start}s")
                    pre_ictal = raw.copy().crop(tmin=pre_start, tmax=start)
                    pre_ictal.plot(title=f"{file} - Pre-Ictal {i+1}")
                    input("Press Enter to continue to Ictal phase...")

                    if save_fig:
                        os.makedirs(fig_dir, exist_ok=True)
                        pre_fig = os.path.join(fig_dir, f"{file[:-4]}_pre_ictal_{i+1}.png")
                        plt.savefig(pre_fig)
                        print(f"Saved pre-ictal plot: {pre_fig}")
                else:
                    print("  Skipping pre-ictal: Not enough data before seizure start.")

                # --- Ictal phase ---
                ictal = raw.copy().crop(tmin=start, tmax=end)
                ictal.plot(title=f"{file} - Ictal {i+1}")
                input("Press Enter to continue to Post-Ictal phase...")

                if save_fig:
                    ictal_fig = os.path.join(fig_dir, f"{file[:-4]}_ictal_{i+1}.png")
                    plt.savefig(ictal_fig)
                    print(f"Saved ictal plot: {ictal_fig}")

                # --- Post-ictal phase ---
                post_end = end + post_ictal_duration
                if post_end <= raw.times[-1]:
                    print(f"  Post-ictal {i+1}: {end}s to {post_end}s")
                    post_ictal = raw.copy().crop(tmin=end, tmax=post_end)
                    post_ictal.plot(title=f"{file} - Post-Ictal {i+1}")
                    input("Press Enter to continue to the next seizure or file...")

                    if save_fig:
                        post_fig = os.path.join(fig_dir, f"{file[:-4]}_post_ictal_{i+1}.png")
                        plt.savefig(post_fig)
                        print(f"Saved post-ictal plot: {post_fig}")
                else:
                    print("  Skipping post-ictal: Not enough data after seizure end.")

# ----------- MAIN EXECUTION -----------
if __name__ == "__main__":
    process_one_patient(
        data_root=data_root,
        summary_file=summary_file,
        save_fig=save_fig,
        fig_dir=fig_dir,
        pre_ictal_duration=pre_ictal_duration,
        post_ictal_duration=post_ictal_duration
    )
