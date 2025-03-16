import os
from sklearn.decomposition import PCA
import scipy.io as sio
import pandas as pd
from brainiak.funcalign.srm import SRM

# Directory containing subject subfolders
DATA_DIR = "./data"  # Path to the directory containing subject folders
OUTPUT_CSV = "brain_activation_SRM.csv"
SHARED_DIM = 100 # number of shared features for SRM

# Load data from a subject's .mat file
def load_subject_data(filepath):
    try:
        mat_data = sio.loadmat(filepath)
        examples = mat_data.get("examples")  # 180 x num_voxels
        keyConcept = mat_data.get("keyConcept").squeeze()  # Get the words
        # Convert words to strings, handling the nested array structure
        words = [w[0] for w in keyConcept]
        return examples, words
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None, None

# Get list of subject subfolders
subject_folders = [os.path.join(DATA_DIR, subject_id) for subject_id in os.listdir(DATA_DIR) 
                  if os.path.isdir(os.path.join(DATA_DIR, subject_id))]
print(f"Found {len(subject_folders)} subject folders")

# List to store each subject's data matrix (shape: 180 x num_voxels)
subject_data = []
subject_words = None
INITIAL_DIM = 180  # Choose a reasonable intermediate dimension

# Initialize PCA
pca = PCA(n_components=INITIAL_DIM)

for subject_folder in subject_folders:
    mat_file_path = os.path.join(subject_folder, "data_180concepts_sentences.mat")
    if not os.path.exists(mat_file_path):
        print(f"Skipping {subject_folder}: data file not found.")
        continue

    examples, words = load_subject_data(mat_file_path)
    if examples is None or words is None:
        continue

    # Print original shape
    print(f"Subject data shape for {subject_folder}: {examples.shape}")
    
    # Apply PCA to reduce dimensionality
    examples_reduced = pca.fit_transform(examples)
    print(f"Reduced shape for {subject_folder}: {examples_reduced.shape}")

    # For consistency, use the words from the first subject
    if subject_words is None:
        subject_words = words
    else:
        if words != subject_words:
            print(f"Warning: words in {subject_folder} do not match the reference. Skipping subject.")
            continue

    subject_data.append(examples_reduced)

print(f"Loaded data from {len(subject_data)} subjects.")

# Run BrainIAK SRM
srm = SRM(n_iter=20, features=SHARED_DIM)
srm.fit(subject_data)
# srm.s_ is the shared response matrix with shape (n_timepoints, SHARED_DIM)
# Here, n_timepoints == 180 (one per concept)
shared_response = srm.s_.T  # Common representation for each word

# Add some basic validation
if shared_response.shape != (180, SHARED_DIM):
    print(f"Warning: Unexpected shape of shared response: {shared_response.shape}")


# Create a DataFrame: rows are words, columns are shared features
df = pd.DataFrame(shared_response, index=subject_words, 
                  columns=[f"srm_{i}" for i in range(SHARED_DIM)])
df.index.name = "word"
df.to_csv(OUTPUT_CSV)

print(f"\nSaved SRM shared response to {OUTPUT_CSV}")
print(f"Final shape: {df.shape}")
