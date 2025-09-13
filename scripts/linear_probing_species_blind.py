import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
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

def load_and_prepare_data():
    """Load and prepare the combined dataset"""
    print("Loading datasets...")
    
    # Load class 0 data
    df1 = pd.read_csv('sampled_vcf_snv_sequences_by_type_embeddings_simple_mean_pool.csv')
    df1['class'] = 0
    df1['species_key'] = df1['VCF_Source'].str.replace('_incl_consequences.vcf.gz', '')
    df1['species_name'] = df1['species_key'].map(lambda x: VCF_INFO.get(x, {}).get('species_name', 'unknown'))
    
    # Load class 1 data
    df2 = pd.read_csv('extracted_flanking_sequences_ref_alt_pyfaidx_hdr_embeddings_simple_mean_pool.csv')
    df2['class'] = 1
    
    # Filter class 1 data
    print(f"Class 1 before filtering: {len(df2)} samples")
    df2 = df2[(df2['Species Name'] != 'rabbit') & (df2['Processing_Status'] == 'Success')]
    print(f"Class 1 after filtering: {len(df2)} samples")
    
    df2['species_name'] = df2['Species Name']
    
    # Identify embedding columns
    ref_cols = [f'Ref_Emb_{i}' for i in range(4096) if f'Ref_Emb_{i}' in df1.columns and f'Ref_Emb_{i}' in df2.columns]
    alt_cols = [f'Alt_Emb_{i}' for i in range(4096) if f'Alt_Emb_{i}' in df1.columns and f'Alt_Emb_{i}' in df2.columns]
    feature_cols = ref_cols + alt_cols
    
    print(f"Feature columns: {len(ref_cols)} Ref + {len(alt_cols)} Alt = {len(feature_cols)} total")
    
    # Combine datasets
    essential_cols = feature_cols + ['species_name', 'class']
    df1_subset = df1[essential_cols].copy()
    df2_subset = df2[essential_cols].copy()
    
    combined_df = pd.concat([df1_subset, df2_subset], ignore_index=True)
    
    # Remove missing values
    print(f"Before removing NaN: {len(combined_df)} samples")
    combined_df = combined_df.dropna(subset=feature_cols)
    print(f"After removing NaN: {len(combined_df)} samples")
    
    # Extract final arrays
    X = combined_df[feature_cols].values
    y = combined_df['class'].values
    species = combined_df['species_name'].values
    
    print(f"\nFinal dataset: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"Class distribution: Class 0: {sum(y == 0)}, Class 1: {sum(y == 1)}")
    print(f"Species distribution:")
    species_counts = pd.Series(species).value_counts()
    for species_name, count in species_counts.items():
        class_0_count = sum((species == species_name) & (y == 0))
        class_1_count = sum((species == species_name) & (y == 1))
        print(f"  {species_name}: {count} total (Class 0: {class_0_count}, Class 1: {class_1_count})")
    
    return X, y, species

def run_species_blind_cv(X, y, species, n_folds=10, C=0.1):
    """Run stratified k-fold cross-validation"""
    print(f"\nRunning {n_folds}-fold stratified cross-validation...")
    print(f"Using Lasso regression with C={C}")
    
    # Initialize cross-validator
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Initialize model
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(penalty='l1', solver='liblinear', 
                                        C=C, max_iter=1000, random_state=42))
    ])
    
    # Storage for results
    results = {
        'fold': [],
        'auroc': [],
        'auprc': [],
        'n_train': [],
        'n_test': [],
        'train_species_dist': [],
        'test_species_dist': []
    }
    
    # Cross-validation loop
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\nFold {fold}:")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        species_train, species_test = species[train_idx], species[test_idx]
        
        print(f"  Train: {len(X_train)} samples, Test: {len(X_test)} samples")
        print(f"  Train classes: {np.bincount(y_train)}")
        print(f"  Test classes: {np.bincount(y_test)}")
        
        # Fit model and predict
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        auroc = roc_auc_score(y_test, y_pred_proba)
        auprc = average_precision_score(y_test, y_pred_proba)
        
        print(f"  AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}")
        
        # Species distributions
        train_species_dist = pd.Series(species_train).value_counts().to_dict()
        test_species_dist = pd.Series(species_test).value_counts().to_dict()
        
        # Store results
        results['fold'].append(fold)
        results['auroc'].append(auroc)
        results['auprc'].append(auprc)
        results['n_train'].append(len(X_train))
        results['n_test'].append(len(X_test))
        results['train_species_dist'].append(train_species_dist)
        results['test_species_dist'].append(test_species_dist)
    
    return results

def analyze_results(results):
    """Analyze and visualize cross-validation results"""
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("SPECIES-BLIND CROSS-VALIDATION RESULTS")
    print("="*60)
    
    # Summary statistics
    auroc_mean, auroc_std = results_df['auroc'].mean(), results_df['auroc'].std()
    auprc_mean, auprc_std = results_df['auprc'].mean(), results_df['auprc'].std()
    
    print(f"\nPerformance Summary:")
    print(f"AUROC: {auroc_mean:.4f} ± {auroc_std:.4f}")
    print(f"AUPRC: {auprc_mean:.4f} ± {auprc_std:.4f}")
    print(f"AUROC range: [{results_df['auroc'].min():.4f}, {results_df['auroc'].max():.4f}]")
    print(f"AUPRC range: [{results_df['auprc'].min():.4f}, {results_df['auprc'].max():.4f}]")
    
    # Detailed fold results
    print(f"\nFold-by-fold Results:")
    display_df = results_df[['fold', 'auroc', 'auprc', 'n_train', 'n_test']].copy()
    print(display_df.to_string(index=False, float_format='%.4f'))
    
    # Analyze species mixing
    print(f"\nSpecies Distribution Analysis:")
    all_species = set()
    for train_dist in results['train_species_dist']:
        all_species.update(train_dist.keys())
    
    species_in_folds = {species: {'train': 0, 'test': 0} for species in all_species}
    
    for fold_idx in range(len(results['fold'])):
        train_dist = results['train_species_dist'][fold_idx]
        test_dist = results['test_species_dist'][fold_idx]
        
        for species in all_species:
            if species in train_dist:
                species_in_folds[species]['train'] += 1
            if species in test_dist:
                species_in_folds[species]['test'] += 1
    
    print("Species presence across folds:")
    for species, counts in species_in_folds.items():
        print(f"  {species}: appears in {counts['train']}/10 train sets, {counts['test']}/10 test sets")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # AUROC by fold
    axes[0, 0].bar(results_df['fold'], results_df['auroc'])
    axes[0, 0].axhline(y=auroc_mean, color='red', linestyle='--', label=f'Mean: {auroc_mean:.3f}')
    axes[0, 0].axhline(y=0.5, color='gray', linestyle=':', alpha=0.7, label='Random')
    axes[0, 0].set_xlabel('Fold')
    axes[0, 0].set_ylabel('AUROC')
    axes[0, 0].set_title('AUROC by Fold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # AUPRC by fold
    axes[0, 1].bar(results_df['fold'], results_df['auprc'])
    axes[0, 1].axhline(y=auprc_mean, color='red', linestyle='--', label=f'Mean: {auprc_mean:.3f}')
    axes[0, 1].set_xlabel('Fold')
    axes[0, 1].set_ylabel('AUPRC')
    axes[0, 1].set_title('AUPRC by Fold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Distribution plots
    axes[1, 0].hist(results_df['auroc'], bins=8, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(x=auroc_mean, color='red', linestyle='--', label=f'Mean: {auroc_mean:.3f}')
    axes[1, 0].set_xlabel('AUROC')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('AUROC Distribution')
    axes[1, 0].legend()
    
    axes[1, 1].hist(results_df['auprc'], bins=8, alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(x=auprc_mean, color='red', linestyle='--', label=f'Mean: {auprc_mean:.3f}')
    axes[1, 1].set_xlabel('AUPRC')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('AUPRC Distribution')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()
    
    return results_df

def main():
    """Main execution function"""
    try:
        # Load and prepare data
        X, y, species = load_and_prepare_data()
        
        # Run cross-validation
        results = run_species_blind_cv(X, y, species, n_folds=10, C=0.1)
        
        # Analyze results
        results_df = analyze_results(results)
        
        # Save results
        results_df.to_csv('species_blind_cv_results.csv', index=False)
        print(f"\nResults saved to 'species_blind_cv_results.csv'")
        
        return results_df
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    results_df = main()

