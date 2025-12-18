from abc import ABC, abstractmethod
import pandas as pd
from sklearn.base import BaseEstimator
from typing import List, Tuple
from sklearn import preprocessing
import numpy as np
from sklearn.impute import SimpleImputer


def fit_scaler(scaling_method: str, df: pd.DataFrame) -> BaseEstimator:
    """

    Parameters
    ----------
    scaling_method : {"MinMax", "Standard"}
        Name of the normalization strategy to apply.
    df : pd.DataFrame
        DataFrame that contains only the continuous features to scale.

    Returns
    -------
    sklearn.base.BaseEstimator
        Scaler instance fitted on `df`.

    """
    if scaling_method == "MinMax":
        return preprocessing.MinMaxScaler().fit(df)
    elif scaling_method == "Standard":
        return preprocessing.StandardScaler().fit(df)
    else:
        raise ValueError(f"Scaling method '{scaling_method}' not known")

def fit_encoder(encoding_method: str, df: pd.DataFrame) -> BaseEstimator:
    """

    Parameters
    ----------
    encoding_method : {"OneHot", "LabelEncoder"}
        Name of the categorical encoder to instantiate.
    df : pd.DataFrame
        DataFrame containing only the categorical columns.

    Returns
    -------
    sklearn.base.BaseEstimator or None
        Fitted encoder; can be None if `df` has no rows.

    """
    if df.empty:
        return None
    if encoding_method == "OneHot":
        return preprocessing.OneHotEncoder(
            handle_unknown="ignore", sparse_output=False
        ).fit(df)
    elif encoding_method == "LabelEncoder":
        return preprocessing.OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        ).fit(df)
    else:
        raise ValueError(f"Encoding method '{encoding_method}' not known")


class Data:
    def __init__(
        self,
        train_path: str,
        eval_path: str,
        categorical: List[str],
        continuous: List[str],
        immutables: List[str],
        target: str,
        encoding_method: str = "OneHot",
        scaling_method: str = "Standard"
    ):
        """

        Parameters
        ----------
        train_path : str
            Path to the training Excel file.
        eval_path : str
            Optional path to the evaluation Excel file.
        categorical : list[str]
            Column names considered categorical.
        continuous : list[str]
            Column names considered continuous.
        immutables : list[str]
            Columns to drop prior to modeling (IDs, labels).
        target : str
            Name of the prediction target.
        encoding_method : str, optional
            Encoder type forwarded to `fit_encoder`.
        scaling_method : str, optional
            Scaler type forwarded to `fit_scaler`.

        """
        self._categorical = categorical
        self._continuous = continuous
        self._immutables = immutables
        self._target = target
        
        # Load training spreadsheet and optionally evaluation spreadsheet
        self._df = pd.read_excel(train_path)
        # Use the same schema for eval by default to prevent downstream errors
        if eval_path:
            self._df_eval = pd.read_excel(eval_path)
        else:
            self._df_eval = pd.DataFrame(columns=self._df.columns)

        # Initialize encoder only when categorical columns are provided
        if self._categorical:
            self._encoder = fit_encoder(encoding_method, self._df[self._categorical])
        else:
            self._encoder = None

        # Initialize scaler only when continuous columns are provided
        if self._continuous:
            self._scaler = fit_scaler(scaling_method, self._df[self._continuous])
        else:
            self._scaler = None

    @property
    def df(self) -> pd.DataFrame:
        """
        Returns
        -------
        pd.DataFrame
            Copy of the training dataframe to avoid in-place edits.
        """
        return self._df.copy()

    def encode(self, features: List[str], df: pd.DataFrame) -> pd.DataFrame:
        """

        Parameters
        ----------
        features : list[str]
            Columns to encode using the fitted encoder.
        df : pd.DataFrame
            DataFrame to transform.

        Returns
        -------
        pd.DataFrame
            DataFrame with encoded categorical columns.

        """
        if not features or self._encoder is None:
            return df
            
        X = df[features]
        if isinstance(self._encoder, preprocessing.OneHotEncoder):
            arr = self._encoder.transform(X)
            cols = self._encoder.get_feature_names_out(features)
            df_enc = pd.DataFrame(arr, columns=cols, index=df.index)
        else:
            arr = self._encoder.transform(X)
            df_enc = pd.DataFrame(arr, columns=features, index=df.index)

        df_rest = df.drop(columns=features)
        return pd.concat([df_rest, df_enc], axis=1)

    def scale(self, features: List[str], df: pd.DataFrame) -> pd.DataFrame:
        """

        Parameters
        ----------
        features : list[str]
            Columns to scale using the fitted scaler.
        df : pd.DataFrame
            DataFrame to transform.

        Returns
        -------
        pd.DataFrame
            DataFrame with scaled continuous columns.

        """
        if not features or self._scaler is None:
            return df
            
        X = df[features]
        arr = self._scaler.transform(X)
        df_scaled = pd.DataFrame(arr, columns=features, index=df.index)

        df_rest = df.drop(columns=features)
        return pd.concat([df_rest, df_scaled], axis=1)

    def reduced_by_multicoll(self, dataframe) -> set:
        """

        Parameters
        ----------
        dataframe : pd.DataFrame
            Dataset on which to compute correlations.

        Returns
        -------
        set
            Feature names flagged for removal due to multicollinearity.

        """
        df = dataframe.copy()
        # Compute the correlation matrix across numeric columns
        corr = df.corr(numeric_only=True)
        threshold = 0.85
        
        high_corr = [
            (corr.columns[i], corr.columns[j], corr.iloc[i, j])
            for i in range(len(corr.columns))
            for j in range(i)
            if abs(corr.iloc[i, j]) > threshold
        ]

        to_remove = set()
        for f1, f2, _ in high_corr:
            # Drop the feature that has lower absolute correlation with the target
            if self._target in corr.columns:
                ct1 = abs(corr.at[f1, self._target])
                ct2 = abs(corr.at[f2, self._target])
                to_remove.add(f1 if ct1 < ct2 else f2)
            else:
                # When the target is missing (e.g., inference), drop the second arbitrarily
                to_remove.add(f2)

        return to_remove

    def reduced_by_low_corr(self, dataframe, threshold_low: float = 0.01) -> List[str]:
        """

        Parameters
        ----------
        dataframe : pd.DataFrame
            Dataset containing the target column.
        threshold_low : float, optional
            Minimum acceptable absolute correlation with the target.

        Returns
        -------
        list[str]
            Features whose correlation is below the threshold.

        """
        df = dataframe.copy()
        if self._target not in df.columns:
            return []
            
        corr_with_target = df.corr(numeric_only=True)[self._target].abs().sort_values()
        # Filtra quelle sotto la soglia, escludendo il target stesso
        low = corr_with_target[corr_with_target <= threshold_low].index.tolist()
        return low


class VesselVerseProcessing(Data):
    """
    Specialized `Data` subclass with feature lists tailored to VesselVerse.
    """

    def __init__(
        self,
        train_path: str,
        eval_path: str,
        train_bool: bool,
        path_csv_saved: str,
        path_eval_csv_saved: str,
        feat_multicoll: list = [],
        feat_low_coll: list = [],
        scaling_method: str = "Standard",
        encoding_method: str = "OneHot",
        save_df: bool = True
    ):
        """

        Parameters
        ----------
        train_path : str
            Excel file containing the training set.
        eval_path : str
            Optional Excel file containing the evaluation set.
        train_bool : bool
            Whether the processor should return (X, y) tuples.
        path_csv_saved : str
            Destination path for the processed train CSV.
        path_eval_csv_saved : str
            Destination path for the processed eval CSV.
        feat_multicoll : list, optional
            Predefined multicollinear features to drop.
        feat_low_coll : list, optional
            Predefined low-correlation features to drop.
        scaling_method : str, optional
            Name of the scaler to use for continuous columns.
        encoding_method : str, optional
            Name of the encoder to use for categorical columns.
        save_df : bool, optional
            Whether to persist processed CSVs to disk.

        """
        self.save_df = save_df
        self.train_bool = train_bool
        self.path_csv_saved = path_csv_saved
        self.path_eval_csv_saved = path_eval_csv_saved
        self.feat_multicoll = feat_multicoll
        self.feat_low_coll = feat_low_coll
        self.scaling_method = scaling_method

        # Categorical features: none in this dataset
        categorical = []

        # Continuous features
        continuous = [
            'total_length', 
            'num_bifurcations', 
            'volume', 
            'num_loops', 
            'num_abnormal_degree_nodes', 
            'Largest_endpoint_root_mean_curvature', 
            #'Largest_endpoint_root_mean_square_curvature', 
            '2nd_Largest_endpoint_root_mean_curvature', 
            #'2nd_Largest_endpoint_root_mean_square_curvature', 
            'Largest_bifurcation_root_mean_curvature', 
            #'Largest_bifurcation_root_mean_square_curvature', 
            'fractal_dimension', 
            'lacunarity', 
            'avg_diameter', 
            'global_mean_loop_length', 
            'global_max_loop_length', 
            'num_components'
        ]

        # Immutable features (decide label1 or label2)
        immutables = ['file_sorgente', 'label1'] 
        # Target feature (label1 or label2)
        target = 'label2'

        super().__init__(
            train_path=train_path,
            eval_path=eval_path,
            categorical=categorical,
            continuous=continuous,
            immutables=immutables,
            target=target,
            encoding_method=encoding_method,
            scaling_method=scaling_method
        )

    def preprocess(self) -> Tuple[pd.DataFrame, pd.Series]:
        """

        Returns
        -------
        Tuple[pd.DataFrame, pd.Series] or pd.DataFrame
            Processed feature matrix and labels when `train_bool` is True;
            otherwise the full processed dataframe.

        Notes
        -----
        Steps performed:
          1. Drop immutable columns.
          2. Impute missing continuous values.
          3. Encode categorical features.
          4. Scale continuous features.
          5. Remove multicollinear and low-correlation features.
          6. Optionally persist processed CSVs.

        """

        df = self.df
        df_eval = self._df_eval.copy()

        # Drop immutables
        df = df.drop(columns=self._immutables, errors='ignore')
        df_eval = df_eval.drop(columns=self._immutables, errors='ignore')

        # Missing Value Imputation
        imputer = SimpleImputer(strategy='mean')
        
        # Fit su train, transform of train e eval
        if self._continuous:
            df[self._continuous] = imputer.fit_transform(df[self._continuous])
            if not df_eval.empty:
                df_eval[self._continuous] = imputer.transform(df_eval[self._continuous])

        # Categorical Encoding
        df = self.encode(self._categorical, df)
        if not df_eval.empty:
            df_eval = self.encode(self._categorical, df_eval)
            # Make sure eval has same columns as train after encoding
            train_cols = [c for c in df.columns if c != self._target]
            df_eval = df_eval.reindex(columns=train_cols, fill_value=0)

        # Scaling Feature Continuous
        df = self.scale(self._continuous, df)
        if not df_eval.empty:
            df_eval = self.scale(self._continuous, df_eval)

        # Feature Selection
        # a) Multicollinearity
        if not self.feat_multicoll:
            to_drop_multi = self.reduced_by_multicoll(df)
            self.feat_multicoll = list(to_drop_multi)
            print(f"Features dropped due to multicollinearity:\n {self.feat_multicoll}")
        
        # b) Low Correlation with target
        if not self.feat_low_coll:
            to_drop_low = self.reduced_by_low_corr(df, threshold_low=0.05) # choose threshold
            self.feat_low_coll = to_drop_low
            print(f"Features dropped due to low correlation:\n {self.feat_low_coll}")

        # Apply feature drops
        cols_to_drop = list(set(self.feat_multicoll + self.feat_low_coll))
        df = df.drop(columns=cols_to_drop, errors='ignore')
        df_eval = df_eval.drop(columns=cols_to_drop, errors='ignore')

        print(f"Final train shape: {df.shape}")
        if not df_eval.empty:
            print(f"Final eval shape: {df_eval.shape}")

        # Save processed data
        if self.save_df:
            df.to_csv(self.path_csv_saved, index=False)
            if not df_eval.empty:
                df_eval.to_csv(self.path_eval_csv_saved, index=False)

        # Return X, y for training or full df for inference
        if self.train_bool:
            X = df.drop(columns=[self._target], errors='ignore')
            y = df[self._target] if self._target in df.columns else None
            return X, y
        else:
            return df

    def get_feat_multicoll(self):
        """Return cached list of multicollinear features dropped."""
        return self.feat_multicoll

    def get_feat_low_coll(self):
        """Return cached list of low-correlation features dropped."""
        return self.feat_low_coll
