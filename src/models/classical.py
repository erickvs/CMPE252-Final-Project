from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC

def build_svm_pipeline(config):
    """
    Builds a PCA + Linear SVM pipeline.
    Note: We use LinearSVC instead of SVC(kernel='rbf') because RBF has O(N^3)
    time complexity, which takes days on the 50,000 CIFAR-10 training samples.
    """
    pca_components = config.get("pca_components", 100)
    c_param = config.get("c_param", 1.0)
    max_iter = config.get("max_iter", 1000)
    
    pipeline = Pipeline([
        ('pca', PCA(n_components=pca_components, random_state=42)),
        ('svm', LinearSVC(C=c_param, max_iter=max_iter, random_state=42, dual='auto'))
    ])
    return pipeline