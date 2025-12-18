import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import (
    ConfusionMatrixDisplay, 
    RocCurveDisplay, 
    PrecisionRecallDisplay
)
from sklearn.decomposition import PCA
from sklearn.base import clone
from matplotlib.colors import ListedColormap


class MLModelClassifier:
    """
    Wrapper that handles splitting, hyper-parameter tuning, evaluation, 
    and visualization for standard scikit-learn classifiers.
    """
    def __init__(
        self,
        X,
        y,
        model_name: str,
        test_size: float = 0.3,
        random_seed: int = 42,
        scoring: str = 'accuracy',
        n_jobs: int = -1,
        verbose: int = 0
    ):
        """
        Parameters
        ----------
        X : array-like
            Feature matrix used for training and validation.
        y : array-like
            Target labels aligned with `X`.
        model_name : str
            Name of the classifier to tune.
        test_size : float, optional
            Fraction of the dataset reserved for the test split.
        random_seed : int, optional
            Seed reused by every stochastic component for reproducibility.
        scoring : str, optional
            Metric optimized inside GridSearchCV.
        n_jobs : int, optional
            Number of parallel jobs for GridSearchCV; -1 means all cores.
        verbose : int, optional
            Verbosity level forwarded to GridSearchCV.
        """
        self.X = X
        self.y = y
        self.model_name = model_name
        self.test_size = test_size
        self.random_seed = random_seed
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.gs = None
        self.X_test = None
        self.y_test = None

    def splitting_data(self):
        """Split (X, y) into train/test sets using the stored ratio and seed."""
        return train_test_split(
            self.X, self.y,
            test_size=self.test_size,
            random_state=self.random_seed
        )

    def _get_model_and_param_grid(self):
        """Return the estimator instance and the hyper-parameter grid to explore."""
        name = self.model_name.lower()

        if name == 'logisticregression':
            model = LogisticRegression(random_state=self.random_seed, max_iter=10000)
            param_grid = {
                'penalty': ['l2', 'l1'],
                'C': [0.01, 0.1, 1, 10, 100],
                'solver': ['liblinear', 'saga']
            }

        elif name == 'ridgeclassifier':
            model = RidgeClassifier(random_state=self.random_seed)
            param_grid = {
                'alpha': [0.01, 0.1, 1, 10, 100],
                'fit_intercept': [True, False]
            }

        elif name == 'svc':
            model = SVC(probability=True, random_state=self.random_seed)
            param_grid = {
                'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto']
            }

        elif name == 'randomforestclassifier':
            model = RandomForestClassifier(random_state=self.random_seed)
            param_grid = {
                'n_estimators': [100, 200],
                'criterion': ['gini', 'entropy'],
                'max_depth': [None, 5, 10],
                'bootstrap': [True, False]
            }

        elif name == 'gradientboostingclassifier':
            model = GradientBoostingClassifier(random_state=self.random_seed)
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5]
            }

        elif name == 'histgradientboostingclassifier':
            model = HistGradientBoostingClassifier(random_state=self.random_seed)
            param_grid = {
                'learning_rate': [0.01, 0.1],
                'max_iter': [100, 200],
                'max_depth': [None, 5]
            }

        elif name == 'knnclassifier':
            model = KNeighborsClassifier()
            param_grid = {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }

        elif name == 'sgdclassifier':
            model = SGDClassifier(random_state=self.random_seed, max_iter=1000, tol=1e-3)
            param_grid = {
                'loss': ['hinge', 'log', 'modified_huber'],
                'penalty': ['l2', 'l1', 'elasticnet'],
                'alpha': [1e-4, 1e-3, 1e-2],
                'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
                'eta0': [1e-3, 1e-2, 1e-1]
            }

        else:
            raise ValueError(f"Model '{self.model_name}' is not supported.")

        return model, param_grid

    def get_model(self):
        """Instantiate GridSearchCV with the estimator, grid, scoring, and CV."""
        model, param_grid = self._get_model_and_param_grid()
        return GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            verbose=self.verbose
        )

    def train(self):
        """
        Run the split, fit GridSearchCV on the training set, and cache test data.

        Returns
        -------
        tuple
            y_pred on the held-out set and the fitted GridSearchCV instance.
        """

        X_train, X_test, y_train, y_test = self.splitting_data()
        self.X_test, self.y_test = X_test, y_test

        self.gs = self.get_model()
        self.gs.fit(X_train, y_train)

        y_pred = self.gs.predict(X_test)
        return y_pred, self.gs

    def evaluate(self):
        """Compute accuracy and weighted F1 score on the cached test split."""
        if self.gs is None or self.X_test is None:
            raise RuntimeError("Model not yet trained. Call `train()` first.")

        y_pred = self.gs.predict(self.X_test)
        return {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'f1_weighted': f1_score(self.y_test, y_pred, average='weighted')
        }
        
    # Plotting methods: confusion matrix, ROC curve, decision boundaries 
    # Use ChatGPT by fixing some code issues
    def plot_confusion_matrix(self, normalize: str = None):
        """
        Plot the confusion matrix for the best estimator selected by GridSearch.

        Parameters
        ----------
        normalize : {'true', 'pred', 'all', None}, optional
            Normalization strategy applied before visualization.
        """
        if self.gs is None:
            raise RuntimeError("Model not trained.")

        best_model = self.gs.best_estimator_

        fig, ax = plt.subplots(figsize=(10, 8))
        ConfusionMatrixDisplay.from_estimator(
            best_model,
            self.X_test,
            self.y_test,
            cmap='Blues',
            normalize=normalize,
            ax=ax
        )
        ax.set_title(f"Confusion Matrix - {self.model_name}")
        plt.show()

    def plot_roc_curve(self):
        """Plot the ROC curve if the estimator exposes probability scores."""
        if self.gs is None:
            raise RuntimeError("Model not trained.")

        best_model = self.gs.best_estimator_
        # Some models (e.g., RidgeClassifier) do not implement predict_proba.
        # Catch the exception or warn the user.
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            RocCurveDisplay.from_estimator(
                best_model,
                self.X_test,
                self.y_test,
                ax=ax
            )
            ax.set_title(f"ROC Curve - {self.model_name}")
            plt.show()
        except Exception as e:
            print(f"Unable to plot ROC for {self.model_name}: {e}")
            print("Tip: Use probability=True for SVC; RidgeClassifier lacks predict_proba.")

    def plot_decision_boundaries(self):
        """
        Reduce data to two dimensions via PCA and visualize decision regions.

        Notes
        -----
        Run PCA to reduce the data to 2 dimensions and plot
        the decision boundaries of the best model found.
        
        NOTE: This trains a “shadow model” on the reduced data only for
        visualization purposes. It helps understand the “shape” of the decision,
        but loses the nuances of the features discarded by PCA.
        """
        if self.gs is None:
            raise RuntimeError("Model not trained. Call `train()` first.")

        # Reduce the data to 2 dimensions with PCA
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(self.X)  # Use the whole dataset X for visualization
        
        # Clone the best model (same hyperparameters) but reset it
        # so we can train it on the 2D data
        best_model = self.gs.best_estimator_
        model_2d = clone(best_model)
        
        # Note: Some models require 1D arrays for y, so use ravel()
        model_2d.fit(X_reduced, self.y)

        # Create a grid (mesh) to color the background
        # Define plot boundaries with a bit of margin
        x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
        y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1
        
        # Grid resolution (step size)
        h = 0.02
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))

        # Predict every point on the grid
        # np.c_ concatenates coordinates to create simulated test points
        Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
        
        # If classes are strings, convert them to numbers for plotting.
        # Use a simple trick when Z is non-numeric:
        if Z.dtype.kind in {'U', 'S', 'O'}:  # strings/objects
            unique_classes = np.unique(self.y)
            class_map = {cls: i for i, cls in enumerate(unique_classes)}
            Z = np.vectorize(class_map.get)(Z)
            # Map y as well for the scatter plot
            y_plot = np.vectorize(class_map.get)(self.y)
        else:
            y_plot = self.y

        Z = Z.reshape(xx.shape)

        # Plotting
        plt.figure(figsize=(10, 8))
        
        # Custom colormaps
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
        cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
        
        # Colored background (decision regions)
        plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6)

        # Actual points (scatter plot)
        scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_plot,
                            cmap=cmap_bold, edgecolor='k', s=20)
        
        plt.title(f"Decision Boundary (PCA 2D Projection)\nModel: {self.model_name}")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend(*scatter.legend_elements(), title="Classes")
        plt.show()
        
        # Variance explained info (how faithful is the 2D reduction?)
        explained_variance = np.sum(pca.explained_variance_ratio_) * 100
        print(f"Note: The 2D plot represents {explained_variance:.2f}% of the original data variance.")