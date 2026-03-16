import matplotlib.pyplot as plt
import torchvision
import numpy as np
from src.adapters.data.cifar10_adapter import CIFAR10Adapter

def visualize_class_distribution(data_dir: str = "../data", dataset=None):
    """
    Visualizes 10 samples per class from the CIFAR-10 dataset.
    This replaces the manual plotting code in the notebook.
    """
    # Load Data
    print("Loading CIFAR-10 Data...")
    adapter = CIFAR10Adapter(data_dir=data_dir)
    
    if dataset is None:
        # Use raw dataset for visualization to get PIL images directly
        raw_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True)
    else:
        raw_dataset = dataset
        
    class_names = adapter.get_class_names()

    # Visualize 10 Samples Per Class
    print("Visualizing Class Distribution...")

    # Group indices by class
    class_indices = {i: [] for i in range(10)}
    for idx, label in enumerate(raw_dataset.targets):
        if len(class_indices[label]) < 10:
            class_indices[label].append(idx)
        if all(len(v) == 10 for v in class_indices.values()):
            break

    # Plot Strips (One figure per class)
    for i in range(10): 
        fig, axes = plt.subplots(1, 10, figsize=(15, 1.5))
        # Title the whole strip
        fig.suptitle(f"{class_names[i].upper()}", fontsize=14, fontweight='bold', x=0.02, ha='left')
        
        for j in range(10): 
            idx = class_indices[i][j]
            img, _ = raw_dataset[idx]
            
            axes[j].imshow(img)
            axes[j].axis('off')
        
        plt.show()

def visualize_pca_concept():
    """
    Generates a synthetic 2D dataset to visually demonstrate what PCA does.
    1. Plots correlated data.
    2. Shows the Principal Components (Eigenvectors).
    3. Shows dimensionality reduction (Projection to 1D).
    """
    from sklearn.decomposition import PCA
    
    print("Generating synthetic 2D data for PCA demonstration...")
    
    # 1. Generate Correlated Data
    np.random.seed(42)
    n_samples = 300
    # Mean = [0,0], Covariance = [[2, 1.5], [1.5, 2]] -> Ellipse tilted at 45 degrees
    X = np.random.multivariate_normal([0, 0], [[3, 2.5], [2.5, 3]], n_samples)
    
    pca = PCA(n_components=2)
    pca.fit(X)
    
    # Get components
    mean = pca.mean_
    # Eigenvectors (directions)
    v1 = pca.components_[0] * np.sqrt(pca.explained_variance_[0])
    v2 = pca.components_[1] * np.sqrt(pca.explained_variance_[1])
    
    # Project data onto PC1 (1D representation)
    X_proj = pca.transform(X) # (n, 2)
    X_restored = pca.inverse_transform(X_proj) # (n, 2) but only using info from PC1 and PC2. 
    # To show 1D reduction, we need to zero out PC2
    X_proj_1d = X_proj.copy()
    X_proj_1d[:, 1] = 0
    X_1d_in_2d = pca.inverse_transform(X_proj_1d)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: The Principal Components
    ax = axes[0]
    ax.scatter(X[:, 0], X[:, 1], alpha=0.4, label="Original Data")
    
    # Draw vectors
    # PC1
    ax.arrow(mean[0], mean[1], v1[0], v1[1], color='red', width=0.1, head_width=0.3, label="PC1 (Max Variance)")
    ax.arrow(mean[0], mean[1], -v1[0], -v1[1], color='red', width=0.1, head_width=0.3)
    
    # PC2
    ax.arrow(mean[0], mean[1], v2[0], v2[1], color='blue', width=0.1, head_width=0.3, label="PC2 (Noise/Low Var)")
    ax.arrow(mean[0], mean[1], -v2[0], -v2[1], color='blue', width=0.1, head_width=0.3)
    
    ax.set_title("Step 1: Finding the 'Main Directions'", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Plot 2: Dimensionality Reduction (Projection)
    ax = axes[1]
    ax.scatter(X[:, 0], X[:, 1], alpha=0.15, color='gray', label="Original Data (Shadow)")
    ax.scatter(X_1d_in_2d[:, 0], X_1d_in_2d[:, 1], alpha=0.8, color='red', label="Projected to 1D (PC1)")
    
    # Draw lines connecting original to projected
    for i in range(20): # Draw just a few lines to avoid clutter
        ax.plot([X[i, 0], X_1d_in_2d[i, 0]], [X[i, 1], X_1d_in_2d[i, 1]], 'k--', alpha=0.2)

    ax.set_title("Step 2: Discarding Noise (Projection to 1D)", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    plt.suptitle("Conceptual Understanding of PCA", fontsize=18)
    plt.show()

def visualize_covariance_calculation():
    """
    Demonstrates the calculation of a Covariance Matrix step-by-step
    using a tiny 2D dataset.
    """
    import pandas as pd
    
    # 1. The Data (3 samples, 2 features)
    # Feature 1: Study Hours
    # Feature 2: Test Score
    X = np.array([
        [2, 50],  # Student A: Slack
        [4, 70],  # Student B: Average
        [6, 90]   # Student C: Studious
    ])
    
    print("--- Step 1: The Raw Data (3 Students, 2 Features) ---")
    print("Columns: [Hours Studied, Test Score]")
    print(X)
    print("\n")
    
    # 2. Compute Mean
    mean = np.mean(X, axis=0)
    print(f"--- Step 2: Compute Mean ---\nMean Vector = {mean} (Average Student)\n")
    
    # 3. Center Data
    X_centered = X - mean
    print("--- Step 3: Center the Data (Subtract Mean) ---")
    print("This shows how each student differs from the average.")
    print(X_centered)
    print("\n")
    
    # 4. Compute Covariance Matrix manually
    # Cov = (X_c.T @ X_c) / (m - 1)
    
    cov_matrix = np.cov(X, rowvar=False)
    
    print("--- Step 4: The Covariance Matrix ---")
    print("Sigma =")
    print(cov_matrix)
    print("\nInterpretation:")
    print(f"Variance(Hours): {cov_matrix[0,0]:.2f} (Spread of study times)")
    print(f"Variance(Score): {cov_matrix[1,1]:.2f} (Spread of scores)")
    print(f"Covariance(Hours, Score): {cov_matrix[0,1]:.2f}")
    print("-> Positive value means they move together! (More study = Higher score)")
    
    # Plot
    plt.figure(figsize=(6, 4))
    plt.scatter(X[:,0], X[:,1], color='blue', s=100, zorder=3)
    plt.plot(X[:,0], X[:,1], color='gray', linestyle='--', alpha=0.5)
    plt.scatter(mean[0], mean[1], color='red', marker='X', s=200, label='Mean', zorder=4)
    plt.title("Visualizing Positive Covariance")
    plt.xlabel("Hours Studied")
    plt.ylabel("Test Score")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

def visualize_pixel_covariance(dataset):
    """
    Visualizes the covariance matrix of the central 8x8 pixels of CIFAR-10 images.
    Helps explain that neighboring pixels are highly correlated.
    """
    import seaborn as sns
    
    print("Computing pixel covariance for central 8x8 patch...")
    
    # 1. Get a batch of images
    # dataset is the raw torchvision dataset
    data = dataset.data # (50000, 32, 32, 3)
    
    # Take first 1000 images, convert to grayscale for simplicity
    # (N, 32, 32)
    gray_data = np.mean(data[:1000], axis=3)
    
    # 2. Crop Center 8x8
    center_x, center_y = 16, 16
    delta = 4 # 8x8
    patch = gray_data[:, center_x-delta:center_x+delta, center_y-delta:center_y+delta] # (N, 8, 8)
    
    # 3. Flatten
    X = patch.reshape(1000, -1) # (1000, 64)
    
    # 4. Standardize (Zero Mean)
    X_centered = X - np.mean(X, axis=0)
    
    # 5. Compute Covariance: (1/m) * X.T @ X
    cov_matrix = np.cov(X_centered, rowvar=False) # (64, 64)
    
    # 6. Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cov_matrix, cmap='viridis')
    plt.title("Covariance Matrix of Central 8x8 Pixels\n(Neighboring pixels are correlated!)", fontsize=14)
    plt.xlabel("Pixel Index (0-63)")
    plt.ylabel("Pixel Index (0-63)")
    plt.show()

def visualize_pca_redundancy():
    """
    Scenario 1: The 'Over-Enthusiastic Intern' (Redundancy)
    Simulates measuring height in both CM and Inches.
    """
    from sklearn.decomposition import PCA
    np.random.seed(42)
    n = 50
    # Height in cm (150 to 190)
    height_cm = np.random.normal(170, 10, n)
    # Height in inches (perfectly correlated)
    height_in = height_cm / 2.54
    
    # Add tiny measurement error so it's not a singular matrix
    height_in += np.random.normal(0, 0.1, n)
    
    X = np.stack([height_cm, height_in], axis=1)
    pca = PCA(n_components=1)
    pca.fit(X)
    X_proj = pca.inverse_transform(pca.transform(X))
    
    plt.figure(figsize=(10, 6))
    plt.scatter(height_cm, height_in, alpha=0.7, label="Student Measurements")
    plt.plot(X_proj[:, 0], X_proj[:, 1], color='red', linestyle='--', linewidth=2, label="PC1 (The 'True' Size)")
    
    plt.title("Scenario 1: The Redundant Data\n(Height in CM vs Inches)", fontsize=14, fontweight='bold')
    plt.xlabel("Height (cm)")
    plt.ylabel("Height (inches)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def visualize_pca_noise():
    """
    Scenario 2: The 'Shaky Hand' (Noise Removal)
    Simulates a drone flying in a straight line but buffeted by wind.
    """
    from sklearn.decomposition import PCA
    np.random.seed(42)
    n = 100
    # True path: Drone moves from (0,0) to (100, 50)
    t = np.linspace(0, 100, n)
    true_x = t
    true_y = 0.5 * t 
    
    # Wind gusts (Noise) - mostly perpendicular to path? 
    # Actually PCA removes low variance. If noise is high variance, PCA keeps noise.
    # Assumption: Signal variance >> Noise variance.
    noise_x = np.random.normal(0, 5, n)
    noise_y = np.random.normal(0, 5, n)
    
    x = true_x + noise_x
    y = true_y + noise_y
    
    X = np.stack([x, y], axis=1)
    pca = PCA(n_components=1)
    pca.fit(X)
    X_proj = pca.inverse_transform(pca.transform(X))
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.5, color='gray', label="Noisy GPS Readings")
    plt.plot(X_proj[:, 0], X_proj[:, 1], color='blue', linewidth=3, label="PC1 (Recovered Flight Path)")
    plt.plot(true_x, true_y, color='green', linestyle=':', linewidth=2, label="True Path")
    
    plt.title("Scenario 2: The Noisy Signal\n(Drone Path + Wind)", fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def visualize_pca_separation_trap():
    """
    Scenario 3: The 'Separation Trap'
    """
    from sklearn.decomposition import PCA
    np.random.seed(42)
    n = 200
    # Two parallel clusters (Hotdog bun shape)
    # Cluster 1: Y ~ 2, X varies a lot
    x1 = np.random.normal(0, 5, n//2)
    y1 = np.random.normal(3, 0.5, n//2)
    
    # Cluster 2: Y ~ -2, X varies a lot
    x2 = np.random.normal(0, 5, n//2)
    y2 = np.random.normal(-3, 0.5, n//2)
    
    x = np.concatenate([x1, x2])
    y = np.concatenate([y1, y2])
    colors = ['blue']*(n//2) + ['orange']*(n//2)
    
    X = np.stack([x, y], axis=1)
    pca = PCA(n_components=1)
    pca.fit(X)
    X_proj = pca.inverse_transform(pca.transform(X))
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, c=colors, alpha=0.6, label="Classes")
    
    # Draw the PC1 line
    plt.plot(X_proj[:, 0], X_proj[:, 1], color='red', linewidth=3, linestyle='--', label="PC1 (Max Variance)")
    
    plt.title("Scenario 3: The Separation Trap\n(Variance is Horizontal, Meaning is Vertical!)", fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # Force equal aspect so the 'thinness' is visible
    plt.axis('equal')
    plt.show()

def visualize_pca_irreducible():
    """
    Scenario 4: Irreducible Data (The Blob)
    """
    from sklearn.decomposition import PCA
    np.random.seed(42)
    n = 300
    x = np.random.normal(0, 3, n)
    y = np.random.normal(0, 3, n)
    
    X = np.stack([x, y], axis=1)
    pca = PCA(n_components=1)
    pca.fit(X)
    X_proj = pca.inverse_transform(pca.transform(X))
    
    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, alpha=0.5, color='purple')
    plt.plot(X_proj[:, 0], X_proj[:, 1], color='red', linestyle='--', linewidth=2, label="PC1")
    
    plt.title("Scenario 4: The Blob (Irreducible)\n(No structure to exploit)", fontsize=14, fontweight='bold')
    plt.legend()
    plt.axis('equal')
    plt.show()

def visualize_scree_plot(dataset):
    """
    Computes PCA on a subset of CIFAR-10 and plots the Scree Plot.
    This answers: "How many dimensions do we actually need?"
    """
    from sklearn.decomposition import PCA
    import numpy as np
    import matplotlib.pyplot as plt

    print("Computing PCA for Scree Plot (using 1000 images)...")
    
    # Flatten images
    X = dataset.data.reshape(50000, -1)[:1000].astype(float) / 255.0
    
    # Fit PCA with enough components to see the curve
    n_components = 100
    pca = PCA(n_components=n_components)
    pca.fit(X)
    
    # Cumulative Variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_components + 1), cumulative_variance, marker='.', linestyle='-', color='b')
    plt.axhline(y=0.90, color='r', linestyle='--', label='90% Variance Explained')
    plt.axhline(y=0.95, color='g', linestyle='--', label='95% Variance Explained')
    
    plt.title("Scree Plot: Cumulative Explained Variance", fontsize=14, fontweight='bold')
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Variance Explained")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

def visualize_pca_scenarios():
    """
    Visualizes 4 conceptual scenarios for PCA:
    1. Perfect Redundancy (Good)
    2. Noise Removal (Good)
    3. The Separation Trap (Bad - Loss of Class Info)
    4. Irreducible Data (Bad - High Loss)
    """
    from sklearn.decomposition import PCA
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # Scenario 1: Perfect Redundancy (Height in cm vs inches)
    # x = random, y = k*x
    np.random.seed(42)
    n = 100
    x1 = np.random.rand(n) * 10
    y1 = x1 * 2.54 # Perfectly correlated
    
    pca1 = PCA(n_components=1)
    X1 = np.stack([x1, y1], axis=1)
    pca1.fit(X1)
    X1_proj = pca1.inverse_transform(pca1.transform(X1))
    
    ax = axes[0, 0]
    ax.scatter(x1, y1, alpha=0.6, label="Data")
    ax.plot(X1_proj[:, 0], X1_proj[:, 1], color='red', linestyle='--', label="PC1 (1D)")
    ax.set_title("1. Obvious Redundancy (Good)\n(e.g., Inches vs. cm)", fontweight='bold')
    ax.set_xlabel("Feature A")
    ax.set_ylabel("Feature B")
    ax.legend()
    
    # Scenario 2: Noise Removal (Signal + Noise)
    # x = signal, y = signal + noise
    x2 = np.linspace(0, 10, n)
    y2 = x2 + np.random.normal(0, 1.5, n)
    
    pca2 = PCA(n_components=1)
    X2 = np.stack([x2, y2], axis=1)
    pca2.fit(X2)
    X2_proj = pca2.inverse_transform(pca2.transform(X2))
    
    ax = axes[0, 1]
    ax.scatter(x2, y2, alpha=0.6, label="Noisy Data")
    ax.plot(X2_proj[:, 0], X2_proj[:, 1], color='red', linewidth=2, label="PC1 (Denoised)")
    ax.set_title("2. Noise Removal (Good)\n(Variance = Signal, Orthogonal = Noise)", fontweight='bold')
    ax.legend()
    
    # Scenario 3: The Separation Trap (Variance != Importance)
    # Two classes, separated by Y, but variance is in X
    x3_c1 = np.random.normal(0, 3, n//2)
    y3_c1 = np.random.normal(2, 0.2, n//2) # Class 1 at Y=2
    
    x3_c2 = np.random.normal(0, 3, n//2)
    y3_c2 = np.random.normal(-2, 0.2, n//2) # Class 2 at Y=-2
    
    x3 = np.concatenate([x3_c1, x3_c2])
    y3 = np.concatenate([y3_c1, y3_c2])
    colors = ['blue']*(n//2) + ['orange']*(n//2)
    
    pca3 = PCA(n_components=1)
    X3 = np.stack([x3, y3], axis=1)
    pca3.fit(X3)
    X3_proj = pca3.inverse_transform(pca3.transform(X3))
    
    ax = axes[1, 0]
    ax.scatter(x3, y3, c=colors, alpha=0.6, label="Classes")
    ax.plot(X3_proj[:, 0], X3_proj[:, 1], color='red', linestyle='--', linewidth=2, label="PC1 Projection")
    ax.set_title("3. The Trap (Bad)\n(Max Variance is NOT Class Separation)", fontweight='bold')
    ax.text(0, 0, "Projecting to Red Line\nMerges Classes!", ha='center', color='darkred', fontweight='bold')
    
    # Scenario 4: Irreducible Data (Circle/Blob)
    # Variance is equal in all directions
    x4 = np.random.normal(0, 2, n)
    y4 = np.random.normal(0, 2, n)
    
    pca4 = PCA(n_components=1)
    X4 = np.stack([x4, y4], axis=1)
    pca4.fit(X4)
    X4_proj = pca4.inverse_transform(pca4.transform(X4))
    
    ax = axes[1, 1]
    ax.scatter(x4, y4, alpha=0.6)
    ax.plot(X4_proj[:, 0], X4_proj[:, 1], color='red', linestyle='--', label="PC1")
    ax.set_title("4. Irreducible Data (Bad)\n(Equal Variance = 50% Loss)", fontweight='bold')
    ax.set_aspect('equal')
    ax.legend()
    
    plt.suptitle("When does Dimensionality Reduction work?", fontsize=16)
    plt.show()
