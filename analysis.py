"""
Compare LLM embeddings to brain activations.
For each of 180 words, we have:
- LLM embeddings: 180 x D
    - Phi-3.5
    - Phi-3.5-vision
    - Llama-3.1
    - Llama-3.2-vision
- Brain activations: 180 x 100
    - reduced by PCA and SRM to 100 components
We want to see if there is a correlation between the LLM embedding and the brain activation.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

# define our models
models = ["phi-3.5", "phi-3.5-vision", "llama-3.1", "llama-3.2-vision"]

# define our data paths
brain_data_path = "brain_activation_SRM.csv"
llm_data_path = "word_embeddings.csv"

def load_glove_data(filepath):
    """
    Load the GloVe embeddings from txt.
    Returns:
        numpy array of shape (n_words=180, embedding_dim)
    """
    # Read the file line by line
    embeddings = []
    with open(filepath, 'r') as f:
        for line in f:
            # Split the line and convert to floats, filtering out empty strings
            values = [float(x) for x in line.strip().split() if x]
            if values:  # only add if we got values
                embeddings.append(values)
    
    return np.array(embeddings)

# load GloVe embeddings
glove_embeddings = load_glove_data("data/glove_embeddings.txt")
print(f"Loaded GloVe embeddings with shape: {glove_embeddings.shape}")

# add GloVe to our normalized embeddings
scaler = StandardScaler()
normalized_glove = scaler.fit_transform(glove_embeddings)


def load_brain_data(filepath):
    """
    Load the brain data from csv.
    returns:
        word_to_brain: dict mapping words to their brain activation vectors (100 x 1)
        brain_matrix: numpy array of shape (n_words=180, n_components=100)
    """
    # load csv into df
    df = pd.read_csv(filepath)

    # get the words
    words = df["word"].values

    # get PCA components
    brain_matrix = df.drop("word", axis=1).values

    # build dict mapping words to brain activations
    word_to_brain = {
        word: vector
        for word, vector in zip(words, brain_matrix)
    }

    return word_to_brain, brain_matrix


def load_llm_data(filepath):
    """
    Load the LLM data from csv.
    returns:
        word_to_embeddings: word -> model -> embedding vector
        model_matrices: dict mapping model name to numpy array of shape (n_words=180, embedding_dim=D)
    """
    # load csv into df
    df = pd.read_csv(filepath)

    # parse string vectors into numpy arrays
    def parse_vector_sring(vector_str):
        return np.array([float(x.strip()) for x in vector_str.split(",")])

    # build dict mapping words to embeddings
    word_to_embeddings = {}
    for _, row in df.iterrows():
        word = row["Word"]
        word_to_embeddings[word] = {
            model: parse_vector_sring(row[model])
            for model in models
        }

    # build dict mapping model name to matrix of shape (n_words=180, embedding_dim=D)
    model_matrices = {
        model: np.stack([word_to_embeddings[word][model] for word in df["Word"].values])
        for model in models
    }

    return word_to_embeddings, model_matrices


def normalize_data(word_to_brain, brain_matrix, word_to_embeddings, model_matrices):
    """
    Normalize both LLM embeddings and brain activation data using StandardScaler.
    
    Args:
        word_to_brain: dict of word -> brain activation vector
        brain_matrix: numpy array (n_words, n_components)
        word_to_embeddings: dict of word -> model -> embedding vector
        model_matrices: dict of model_name -> numpy array (n_words, embedding_dim)
    """
    # heck that we have the same words in both datasets
    brain_words = set(word_to_brain.keys())
    embedding_words = set(word_to_embeddings.keys())
    
    if brain_words != embedding_words:
        missing_in_brain = embedding_words - brain_words
        missing_in_embeddings = brain_words - embedding_words
        raise ValueError(
            f"Word mismatch!\n"
            f"Words in embeddings but not brain: {missing_in_brain if missing_in_brain else 'none'}\n"
            f"Words in brain but not embeddings: {missing_in_embeddings if missing_in_embeddings else 'none'}"
        )
    
    # get words in a fixed order (e.g., sorted)
    words = sorted(brain_words)
    
    # verify matrix shapes match our word count
    n_words = len(words)
    if brain_matrix.shape[0] != n_words:
        raise ValueError(f"Brain matrix has {brain_matrix.shape[0]} rows but we have {n_words} words")
    
    for model, matrix in model_matrices.items():
        if matrix.shape[0] != n_words:
            raise ValueError(f"Model {model} matrix has {matrix.shape[0]} rows but we have {n_words} words")
    
    # now proceed with normalization
    scaler = StandardScaler()
    
    # normalize each model's embeddings
    normalized_embeddings = {
        model: scaler.fit_transform(embeddings)
        for model, embeddings in model_matrices.items()
    }
    
    # normalize brain activations
    normalized_brain = scaler.fit_transform(brain_matrix)
    
    # update the word-to-embedding dictionary with normalized vectors
    norm_word_to_embeddings = {}
    for i, word in enumerate(words):  # use our fixed word order
        norm_word_to_embeddings[word] = {
            model: normalized_embeddings[model][i]
            for model in model_matrices.keys()
        }
    
    # update the word-to-brain dictionary with normalized vectors
    norm_word_to_brain = {
        word: normalized_brain[i]
        for i, word in enumerate(words)  # use same word order
    }
    
    return normalized_embeddings, normalized_brain, norm_word_to_embeddings, norm_word_to_brain


normalized_embeddings, normalized_brain, norm_word_to_embeddings, norm_word_to_brain = normalize_data(
    *load_brain_data(brain_data_path),
    *load_llm_data(llm_data_path)
)
normalized_embeddings["glove"] = normalized_glove

# try without normalization
# norm_word_to_brain, norm_brain_matrix = load_brain_data(brain_data_path)

def compute_rdm(data):
    """
    Compute the representational dissimilarity matrix (RDM) for the given data.
    args: data of shape (n_words, n_features)
    returns: rdm of shape (n_words, n_words)
    """
    rdm = squareform(pdist(data, metric="correlation"))
    return rdm


def rsa_analysis(normalized_brain_matrix, normalized_model_matrices):
    """
    Perform an RSA analysis on the normalized brain and model matrices.
    """
    # compute the RDMs
    brain_rdm = compute_rdm(normalized_brain_matrix)
    print(f"Brain RDM shape: {brain_rdm.shape}")
    # compute the RDMs for the model embeddings
    results = {}
    for model_name, embeddings in normalized_model_matrices.items():
        print(f"Computing RDM for {model_name}, shape: {embeddings.shape}")

        # get the current model's RDM
        model_rdm = compute_rdm(embeddings)
        print(f"Model RDM shape: {model_rdm.shape}")

        # get upper triangular parts, ignoring diagonal
        brain_upper = brain_rdm[np.triu_indices(brain_rdm.shape[0], k=1)]
        model_upper = model_rdm[np.triu_indices(model_rdm.shape[0], k=1)]

        print(f"Brain RDM shape: {brain_upper.shape}, Model RDM shape: {model_upper.shape}")

        # compute the Pearson correlation coefficient
        corr_coeff, p_value = pearsonr(brain_upper, model_upper)
        results[model_name] = {
            "correlation": corr_coeff,
            "p_value": p_value
        }

        print(f"Correlation: {corr_coeff}, p-value: {p_value}")
        print()

    return results

rsa_analysis(normalized_brain, normalized_embeddings)


def per_word_regression(normalized_embeddings, normalized_brain, word_to_embeddings):
    """
    Compute regression scores between embeddings and brain activations for each word.
    
    Args:
        normalized_embeddings: dict of model_name -> numpy array (n_words, embedding_dim)
        normalized_brain: numpy array (n_words, n_components)
        word_to_embeddings: dict of word -> model -> embedding vector
    Returns:
        word_scores: dict of word -> model -> regression score
    """
    # Get list of words in a fixed order
    words = sorted(word_to_embeddings.keys())
    
    # Store scores for each word
    word_scores = {}
    
    # Initialize Ridge regression
    # reg = Ridge(alpha=1.0)
    
    for i, word in enumerate(words):
        word_scores[word] = {}
        
        # Get brain activation for this word
        brain_vector = normalized_brain[i]  # shape: (n_brain_components,)
        
        # Compute correlation for each model
        for model in normalized_embeddings.keys():
            model_vector = normalized_embeddings[model][i]  # shape: (embedding_dim,)
            
            # Compute correlation between brain pattern and model embedding
            correlation, p_value = pearsonr(brain_vector, model_vector[:len(brain_vector)])
            
            word_scores[word][model] = {
                'correlation': correlation,
                'p_value': p_value
            }
    
    # Print summary statistics
    print("\nModel-wise summary:")
    for model in normalized_embeddings.keys():
        correlations = [word_scores[word][model]['correlation'] for word in words]
        print(f"\n{model}:")
        print(f"Mean correlation: {np.mean(correlations):.3f} ± {np.std(correlations):.3f}")
        print(f"Range: [{np.min(correlations):.3f}, {np.max(correlations):.3f}]")
        
        # Count significant correlations (p < 0.05)
        sig_count = sum(word_scores[word][model]['p_value'] < 0.05 for word in words)
        print(f"Significant correlations: {sig_count}/{len(words)}")
    
    return word_scores

def plot_word_scores(word_scores):
    """Plot distribution of correlations for each model."""
    # Extract correlations for each model
    models = list(next(iter(word_scores.values())).keys())
    model_scores = {
        model: [word_scores[word][model]['correlation'] 
                for word in word_scores.keys()]
        for model in models
    }
    
    # Create violin plot
    plt.figure(figsize=(10, 6))
    plt.violinplot([model_scores[model] for model in models])
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    plt.xticks(range(1, len(models) + 1), models, rotation=45)
    plt.title('Distribution of Word-Level Correlations')
    plt.ylabel('Correlation')
    plt.tight_layout()
    return plt.gcf()

def print_extreme_scores(word_scores, n=5):
    """Print words with highest and lowest correlations for each model."""
    models = list(next(iter(word_scores.values())).keys())
    
    for model in models:
        print(f"\n{model}:")
        
        # Sort words by correlation
        sorted_words = sorted(
            word_scores.keys(),
            key=lambda w: word_scores[w][model]['correlation']
        )
        
        print("\nBest correlated words:")
        for word in sorted_words[-n:]:
            corr = word_scores[word][model]['correlation']
            p_val = word_scores[word][model]['p_value']
            print(f"{word}: r={corr:.3f} (p={p_val:.3f})")
            
        print("\nWorst correlated words:")
        for word in sorted_words[:n]:
            corr = word_scores[word][model]['correlation']
            p_val = word_scores[word][model]['p_value']
            print(f"{word}: r={corr:.3f} (p={p_val:.3f})")


word_scores = per_word_regression(normalized_embeddings, normalized_brain, norm_word_to_embeddings)
plot_word_scores(word_scores)
plt.savefig("word_scores.png")
plt.show()
print_extreme_scores(word_scores)

def compute_similarities(normalized_embeddings, normalized_brain):
    """
    Compute cosine similarity and MSE between LLM embeddings and brain activations,
    using linear regression to map embeddings to brain space.
    
    Args:
        normalized_embeddings: dict of model_name -> numpy array (n_words, embedding_dim)
        normalized_brain: numpy array (n_words, n_components)
    Returns:
        results: dict of model -> {'cosine': score, 'mse': score}
    """
    results = {}
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for model_name, embeddings in normalized_embeddings.items():
        cosine_sims = []
        mse_scores = []
        
        # Use cross-validation to avoid overfitting
        for train_idx, test_idx in kf.split(embeddings):
            # Train regression model to map embeddings to brain space
            reg = Ridge(alpha=1.0)
            reg.fit(embeddings[train_idx], normalized_brain[train_idx])
            
            # Project test embeddings to brain space
            projected_embeddings = reg.predict(embeddings[test_idx])
            brain_test = normalized_brain[test_idx]
            
            # Compute cosine similarity for each word
            fold_cosine = np.array([
                np.dot(e, b) / (np.linalg.norm(e) * np.linalg.norm(b))
                for e, b in zip(projected_embeddings, brain_test)
            ])
            cosine_sims.extend(fold_cosine)
            
            # Compute MSE
            fold_mse = np.mean((projected_embeddings - brain_test) ** 2)
            mse_scores.append(fold_mse)
        
        # Convert to numpy array for statistics
        cosine_sims = np.array(cosine_sims)
        mean_cosine = np.mean(cosine_sims)
        mean_mse = np.mean(mse_scores)
        
        results[model_name] = {
            'cosine': mean_cosine,
            'mse': mean_mse
        }
        
        print(f"\n{model_name}:")
        print(f"Mean Cosine Similarity: {mean_cosine:.3f}")
        print(f"Mean MSE: {mean_mse:.3f}")
        print(f"Cosine Similarity std: {np.std(cosine_sims):.3f}")
        print(f"Cosine Similarity range: [{np.min(cosine_sims):.3f}, {np.max(cosine_sims):.3f}]")
    
    return results

# Run the analysis
similarity_results = compute_similarities(normalized_embeddings, normalized_brain)

def compute_rank_accuracy(normalized_embeddings, normalized_brain, k_values=[1, 5, 10]):
    """
    Compute rank accuracy between LLM embeddings and brain activations for multiple k values.
    For each k, computes accuracy of whether the k most similar items in embedding space
    match the k most similar items in brain space.
    
    Args:
        normalized_embeddings: dict of model_name -> numpy array (n_words, embedding_dim)
        normalized_brain: numpy array (n_words, n_components)
        k_values: list of k values to evaluate
    Returns:
        results: dict of model -> dict of k -> accuracy score
    """
    results = {}
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for model_name, embeddings in normalized_embeddings.items():
        results[model_name] = {k: [] for k in k_values}
        
        for train_idx, test_idx in kf.split(embeddings):
            # Train regression model to map embeddings to brain space
            reg = Ridge(alpha=1.0)
            reg.fit(embeddings[train_idx], normalized_brain[train_idx])
            
            # Project test embeddings to brain space
            projected_test = reg.predict(embeddings[test_idx])
            brain_test = normalized_brain[test_idx]
            
            # For each word in test set
            for i in range(len(test_idx)):
                # Get similarities to all other test words
                brain_sims = np.array([
                    np.dot(brain_test[i], brain_test[j]) / 
                    (np.linalg.norm(brain_test[i]) * np.linalg.norm(brain_test[j]))
                    for j in range(len(test_idx))
                ])
                model_sims = np.array([
                    np.dot(projected_test[i], projected_test[j]) / 
                    (np.linalg.norm(projected_test[i]) * np.linalg.norm(projected_test[j]))
                    for j in range(len(test_idx))
                ])
                
                # For each k value
                for k in k_values:
                    # Get top k most similar items (excluding self)
                    brain_top_k = set(np.argsort(brain_sims)[-(k+1):-1])  # exclude self
                    model_top_k = set(np.argsort(model_sims)[-(k+1):-1])  # exclude self
                    
                    # Compute overlap
                    overlap = len(brain_top_k & model_top_k)
                    accuracy = overlap / k
                    results[model_name][k].append(accuracy)
        
        # Compute mean and std for each k
        print(f"\n{model_name}:")
        for k in k_values:
            mean_acc = np.mean(results[model_name][k])
            std_acc = np.std(results[model_name][k])
            print(f"k={k:2d}: Accuracy = {mean_acc:.3f} ± {std_acc:.3f}")
    
    return results


# Run the rank analysis
rank_results = compute_rank_accuracy(normalized_embeddings, normalized_brain)

# Plot rank accuracy results for different k values
plt.figure(figsize=(12, 6))
x = np.arange(len(models) + 1)  # +1 for glove
width = 0.15  # width of bars
k_values = list(rank_results[models[0]].keys())

# Calculate random baseline for each k
n_words = normalized_brain.shape[0]  # Total number of words
random_baselines = [k/(n_words-1) for k in k_values]  # -1 to exclude self

# Plot bars for each k value
for i, k in enumerate(k_values):
    accuracies = [np.mean(rank_results[model][k]) for model in models + ["glove"]]
    plt.bar(x + i*width, accuracies, width, label=f'k={k}')
    
    # Plot random baseline as dashed line
    plt.axhline(y=random_baselines[i], color=f'C{i}', linestyle='--', alpha=0.5)

plt.xlabel('Model')
plt.ylabel('Top-k Accuracy')
plt.title('Rank Accuracy by Model and k (dashed lines show random baseline)')
plt.xticks(x + width * (len(k_values)-1)/2, models + ["glove"], rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig("rank_accuracy.png")
plt.show()


