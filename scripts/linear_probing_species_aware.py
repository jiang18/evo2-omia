import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Species mapping dictionary
VCF_INFO = {
    'capra_hircus': {'species_name': 'goat', 'fasta_prefix': 'Capra_hircus'},
    'bos_taurus': {'species_name': 'taurine cattle', 'fasta_prefix': 'Bos_taurus'},
    'canis_lupus_familiaris': {'species_name': 'dog', 'fasta_prefix': 'Canis_lupus_familiaris'},
    'equus_caballus': {'species_name': 'horse', 'fasta_prefix': 'Equus_caballus'},
    'felis_catus': {'species_name': 'domestic cat', 'fasta_prefix': 'Felis_catus'},
    'gallus_gallus': {'species_name': 'chicken', 'fasta_prefix': 'Gallus_gallus'},
    'ovis_aries_rambouillet': {'species_name': 'sheep', 'fasta_prefix': 'Ovis_aries_rambouillet'},
    'sus_scrofa': {'species_name': 'pig', 'fasta_prefix': 'Sus_scrofa'}
}

def load_and_preprocess_data():
    """
    Load and preprocess both datasets
    """
    print("Loading datasets...")
    
    # Load first file (class 0)
    df1 = pd.read_csv('sampled_vcf_snv_sequences_by_type_embeddings_simple_mean_pool.csv')
    df1['class'] = 0
    
    # Extract species from VCF_Source
    df1['species_key'] = df1['VCF_Source'].str.replace('_incl_consequences.vcf.gz', '')
    df1['species_name'] = df1['species_key'].map(lambda x: VCF_INFO.get(x, {}).get('species_name', 'unknown'))
    
    # Load second file (class 1)
    df2 = pd.read_csv('extracted_flanking_sequences_ref_alt_pyfaidx_hdr_embeddings_simple_mean_pool.csv')
    df2['class'] = 1
    
    # Filter out rabbit and failed processing
    print(f"Before filtering: {len(df2)} rows")
    df2 = df2[(df2['Species Name'] != 'rabbit') & (df2['Processing_Status'] == 'Success')]
    print(f"After filtering: {len(df2)} rows removed rabbit and non-success")
    
    df2['species_name'] = df2['Species Name']
    
    print(f"Dataset 1 (class 0): {len(df1)} samples")
    print(f"Dataset 2 (class 1): {len(df2)} samples")
    
    # Check species distribution
    print("\nSpecies distribution:")
    print("Class 0:", df1['species_name'].value_counts().to_dict())
    print("Class 1:", df2['species_name'].value_counts().to_dict())
    
    return df1, df2

def prepare_features_and_labels(df1, df2):
    """
    Prepare feature matrix and labels for classification
    """
    # Identify embedding columns
    ref_emb_cols = [f'Ref_Emb_{i}' for i in range(4096)]
    alt_emb_cols = [f'Alt_Emb_{i}' for i in range(4096)]
    
    # Check which columns actually exist
    available_ref_cols = [col for col in ref_emb_cols if col in df1.columns and col in df2.columns]
    available_alt_cols = [col for col in alt_emb_cols if col in df1.columns and col in df2.columns]
    
    print(f"Available Ref embedding columns: {len(available_ref_cols)}")
    print(f"Available Alt embedding columns: {len(available_alt_cols)}")
    
    feature_cols = available_ref_cols + available_alt_cols
    
    # Combine datasets
    df1_features = df1[feature_cols + ['species_name', 'class']].copy()
    df2_features = df2[feature_cols + ['species_name', 'class']].copy()
    
    combined_df = pd.concat([df1_features, df2_features], ignore_index=True)
    
    # Remove any rows with missing values in features
    print(f"Before removing NaN: {len(combined_df)} samples")
    combined_df = combined_df.dropna(subset=feature_cols)
    print(f"After removing NaN: {len(combined_df)} samples")
    
    X = combined_df[feature_cols].values
    y = combined_df['class'].values
    species = combined_df['species_name'].values
    
    print(f"Final feature matrix shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    return X, y, species, feature_cols

class SpeciesAwareCV:
    """
    Custom cross-validation that keeps species together
    """
    def __init__(self, species_labels):
        self.species_labels = species_labels
        self.unique_species = np.unique(species_labels)
        
    def split(self, X, y=None, groups=None):
        """
        Generate train/test splits using leave-one-species-out
        """
        for test_species in self.unique_species:
            test_idx = np.where(self.species_labels == test_species)[0]
            train_idx = np.where(self.species_labels != test_species)[0]
            
            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self.unique_species)

def tune_hyperparameters(X_train, y_train, species_train, cv_folds=3):
    """
    Tune hyperparameters using nested cross-validation - Lasso only
    """
    print("Tuning Lasso hyperparameters...")
    
    # Create pipeline with scaling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000, random_state=42))
    ])
    
    # Parameter grid for Lasso only
    param_grid = {
        'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]
    }
    
    # Use stratified k-fold for inner CV (not species-aware for hyperparameter tuning)
    inner_cv = StratifiedKFold(n_splits=min(cv_folds, np.min(np.bincount(y_train))), 
                               shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=inner_cv,
        scoring='average_precision',  # AUPRC
        n_jobs=-1,
        verbose=0  # Reduced verbosity
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best C parameter: {grid_search.best_params_['classifier__C']}")
    print(f"Best CV score (AUPRC): {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def evaluate_model():
    """
    Main evaluation function
    """
    # Load and preprocess data
    df1, df2 = load_and_preprocess_data()
    X, y, species, feature_cols = prepare_features_and_labels(df1, df2)
    
    # Initialize results storage
    results = {
        'test_species': [],
        'auroc': [],
        'auprc': [],
        'n_train': [],
        'n_test': [],
        'class_0_train': [],
        'class_1_train': [],
        'class_0_test': [],
        'class_1_test': []
    }
    
    # Species-aware cross-validation
    species_cv = SpeciesAwareCV(species)
    
    print(f"\nPerforming leave-one-species-out cross-validation...")
    print(f"Total species: {len(np.unique(species))}")
    
    fold = 0
    for train_idx, test_idx in species_cv.split(X, y):
        fold += 1
        test_species = species[test_idx][0]  # All test samples have same species
        
        print(f"\nFold {fold}: Testing on {test_species}")
        print(f"Train size: {len(train_idx)}, Test size: {len(test_idx)}")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Check class distribution
        train_class_counts = np.bincount(y_train)
        test_class_counts = np.bincount(y_test)
        
        print(f"Train class distribution: {train_class_counts}")
        print(f"Test class distribution: {test_class_counts}")
        
        # Skip if not enough samples of each class in training
        if len(train_class_counts) < 2 or np.min(train_class_counts) < 2:
            print(f"Skipping fold due to insufficient class representation in training")
            continue
            
        # Tune hyperparameters on training set
        try:
            best_model = tune_hyperparameters(X_train, y_train, species[train_idx])
            
            # Make predictions
            y_pred_proba = best_model.predict_proba(X_test)
            
            # Handle case where only one class is present in test set
            if len(test_class_counts) == 2 and np.min(test_class_counts) > 0:
                auroc = roc_auc_score(y_test, y_pred_proba[:, 1])
                auprc = average_precision_score(y_test, y_pred_proba[:, 1])
            else:
                print("Cannot compute AUROC/AUPRC - only one class in test set")
                auroc = np.nan
                auprc = np.nan
            
            # Store results
            results['test_species'].append(test_species)
            results['auroc'].append(auroc)
            results['auprc'].append(auprc)
            results['n_train'].append(len(train_idx))
            results['n_test'].append(len(test_idx))
            results['class_0_train'].append(train_class_counts[0] if len(train_class_counts) > 0 else 0)
            results['class_1_train'].append(train_class_counts[1] if len(train_class_counts) > 1 else 0)
            results['class_0_test'].append(test_class_counts[0] if len(test_class_counts) > 0 else 0)
            results['class_1_test'].append(test_class_counts[1] if len(test_class_counts) > 1 else 0)
            
            print(f"AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}")
            
        except Exception as e:
            print(f"Error in fold {fold}: {e}")
            continue
    
    return results

def summarize_results(results):
    """
    Summarize and visualize results
    """
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("SPECIES-AWARE CROSS-VALIDATION RESULTS")
    print("="*60)
    
    print(f"\nDetailed Results:")
    print(results_df.to_string(index=False, float_format='%.4f'))
    
    # Calculate summary statistics (excluding NaN values)
    valid_auroc = results_df['auroc'].dropna()
    valid_auprc = results_df['auprc'].dropna()
    
    if len(valid_auroc) > 0:
        print(f"\nSummary Statistics:")
        print(f"Mean AUROC: {valid_auroc.mean():.4f} ± {valid_auroc.std():.4f}")
        print(f"Mean AUPRC: {valid_auprc.mean():.4f} ± {valid_auprc.std():.4f}")
        print(f"Valid folds: {len(valid_auroc)}/{len(results_df)}")
    
    # Plot results if we have valid data
    if len(valid_auroc) > 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # AUROC plot
        ax1.bar(range(len(results_df)), results_df['auroc'])
        ax1.set_xlabel('Test Species')
        ax1.set_ylabel('AUROC')
        ax1.set_title('AUROC by Test Species')
        ax1.set_xticks(range(len(results_df)))
        ax1.set_xticklabels(results_df['test_species'], rotation=45)
        ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Random')
        ax1.legend()
        
        # AUPRC plot
        ax2.bar(range(len(results_df)), results_df['auprc'])
        ax2.set_xlabel('Test Species')
        ax2.set_ylabel('AUPRC')
        ax2.set_title('AUPRC by Test Species')
        ax2.set_xticks(range(len(results_df)))
        ax2.set_xticklabels(results_df['test_species'], rotation=45)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    return results_df

# Main execution
if __name__ == "__main__":
    try:
        results = evaluate_model()
        results_df = summarize_results(results)
        
        # Save results
        results_df.to_csv('species_cv_results.csv', index=False)
        print(f"\nResults saved to 'species_cv_results.csv'")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()

