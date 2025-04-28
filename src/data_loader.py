import os
import numpy as np
from torch.utils.data import Dataset
from scipy.io import loadmat
from .augment import prep_data, random_scale, time_warp

NUM_SUBJECTS = 27
NUM_CHANNELS = 12
WINDOW_SIZE = 200
STRIDE = 50
NUM_CLASSES = 49
TOTAL_CLASSES = NUM_CLASSES + 1
EXERCISES = ['E1', 'E2', 'E3']

class EMGDataset(Dataset):
    def __init__(self, segments, labels, augment=False):
        self.segments = segments
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        segment = self.segments[idx].copy()
        if self.augment:
            segment = random_scale(segment)
            if np.random.rand() > 0.5:
                segment = time_warp(segment)
        return torch.FloatTensor(segment), torch.LongTensor([self.labels[idx]])

def load_and_preprocess_data(data_root):
    all_segments = []
    all_labels = []
    rest_stats = []

    for subject_id in range(1, NUM_SUBJECTS + 1):
        print(f"Processing subject {subject_id}")
        base_path = os.path.join(data_root, f'DB2_s{subject_id}')        subject_segments = []
        subject_segment_labels = []

        for exercise in EXERCISES:
            file_path = os.path.join(base_path, f'S{subject_id}_{exercise}_A1.mat')
            try:
                mat_data = loadmat(file_path)
                emg = mat_data['emg']
                stimulus = mat_data['stimulus'].flatten()

                min_len = min(emg.shape[0], stimulus.shape[0])
                emg = emg[:min_len]
                stimulus = stimulus[:min_len]

                emg = prep_data(emg)  # now properly defined

                changes = np.where(np.diff(stimulus) != 0)[0] + 1
                starts = np.insert(changes, 0, 0)
                ends = np.append(changes, len(stimulus))

                for i in range(len(starts)):
                    start_idx = starts[i]
                    end_idx = ends[i]
                    segment_type = "rest" if stimulus[start_idx] == 0 else "gesture"

                    if segment_type == "rest":
                        rest_length = end_idx - start_idx
                        keep_length = rest_length // 3
                        rest_stats.append({
                            'subject': subject_id,
                            'exercise': exercise,
                            'total_rest': rest_length,
                            'kept_rest': keep_length,
                            'percentage': keep_length/rest_length if rest_length > 0 else 0
                        })
                        if keep_length > WINDOW_SIZE:
                            middle_start = start_idx + (rest_length - keep_length) // 2
                            segment_emg = emg[middle_start:middle_start+keep_length]
                            segment_labels = stimulus[middle_start:middle_start+keep_length]
                        else:
                            continue
                    else:
                        segment_emg = emg[start_idx:end_idx]
                        segment_labels = stimulus[start_idx:end_idx]

                    num_samples = segment_emg.shape[0]
                    for window_start in range(0, num_samples - WINDOW_SIZE, STRIDE):
                        window_end = window_start + WINDOW_SIZE
                        window_emg = segment_emg[window_start:window_end]
                        window_labels = segment_labels[window_start:window_end]

                        if np.all(window_labels == 0):
                            label = NUM_CLASSES
                        else:
                            gesture_labels = window_labels[window_labels != 0]
                            label = np.argmax(np.bincount(gesture_labels)) if len(gesture_labels) > 0 else NUM_CLASSES

                        subject_segments.append(window_emg)
                        subject_segment_labels.append(label)

            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")

        if subject_segments:
            all_segments.extend(subject_segments)
            all_labels.extend(subject_segment_labels)

    if rest_stats:
        total_rest = sum(s['total_rest'] for s in rest_stats)
        kept_rest = sum(s['kept_rest'] for s in rest_stats)
        avg_percentage = np.mean([s['percentage'] for s in rest_stats])
        print(f"\nRest Period Statistics:")
        print(f"Total rest samples: {total_rest}")
        print(f"Kept rest samples: {kept_rest}")
        print(f"Average percentage kept: {avg_percentage:.1%}")
        print(f"Number of rest segments processed: {len(rest_stats)}")

    return np.array(all_segments), np.array(all_labels)
