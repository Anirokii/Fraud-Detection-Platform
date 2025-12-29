"""
Fraud Detection Model Training
===============================
Ce script entraîne un modèle XGBoost pour détecter les fraudes,
en utilisant SMOTE pour le class imbalance et MLflow pour le tracking.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import time
import warnings
warnings.filterwarnings('ignore')
import tempfile
import shutil

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, roc_curve
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# MLflow
import mlflow
import mlflow.xgboost

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
MLFLOW_DIR = BASE_DIR / "mlflow"

# Créer les dossiers
MODELS_DIR.mkdir(exist_ok=True)
MLFLOW_DIR.mkdir(exist_ok=True)

# Configuration MLflow
mlflow.set_tracking_uri(f"file:///{MLFLOW_DIR}")
mlflow.set_experiment("fraud-detection")


class FraudModelTrainer:
    """Classe pour entraîner le modèle de détection de fraude"""
    
    def __init__(self):
        """Initialisation"""
        print("="*70)
        print("🚀 FRAUD DETECTION MODEL TRAINING")
        print("="*70)
        
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = None
        
    def load_data(self):
        """
        Étape 1: Charger les données enrichies
        """
        print("\n📊 Chargement des données...")
        
        data_path = DATA_DIR / "fraud_transactions_enriched.csv"
        self.df = pd.read_csv(data_path)
        
        print(f"✅ {len(self.df):,} transactions chargées")
        print(f"   - Fraudes: {self.df['is_fraud'].sum():,} ({self.df['is_fraud'].mean()*100:.3f}%)")
        print(f"   - Légitimes: {(~self.df['is_fraud'].astype(bool)).sum():,}")
        
        return self
    
    def prepare_features(self):
        """
        Étape 2: Préparer les features pour le ML
        
        FEATURES SÉLECTIONNÉES:
        - Temporelles: hour, day_of_week, is_weekend, is_night
        - Comportementales: tx_velocity, amt_deviation, customer_tx_count, days_since_first_tx
        - Catégorielles: category, gender, state
        - Montant: amt
        - Risque: is_high_risk_category
        """
        print("\n🔧 Préparation des features...")
        
        # Features numériques
        numeric_features = [
            'amt',                    # Montant
            'hour',                   # Heure
            'day_of_week',           # Jour semaine
            'is_weekend',            # Weekend
            'is_night',              # Nuit
            'tx_velocity',           # Vélocité
            'amt_deviation',         # Déviation montant
            'customer_tx_count',     # Nb transactions
            'days_since_first_tx',   # Jours depuis 1ère tx
            'is_high_risk_category', # Catégorie risque
            'customer_age'           # Âge client
        ]
        
        # Features catégorielles à encoder
        categorical_features = [
            'category',  # Catégorie marchand
            'gender',    # Genre client
            'state'      # État
        ]
        
        # Créer le dataset
        X = self.df[numeric_features + categorical_features].copy()
        y = self.df['is_fraud'].values
        
        # Gérer les valeurs manquantes
        X = X.fillna(0)
        
        # Encoder les variables catégorielles
        print("  🏷️  Encodage des variables catégorielles...")
        for col in categorical_features:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        # Sauvegarder les noms de features
        self.feature_names = X.columns.tolist()
        
        print(f"✅ {len(self.feature_names)} features préparées:")
        for feat in self.feature_names:
            print(f"   • {feat}")
        
        self.X = X
        self.y = y
        
        return self
    
    def split_data(self, test_size=0.3, random_state=42):
        """
        Étape 3: Séparer train/test
        """
        print(f"\n✂️  Split des données (train: {(1-test_size)*100:.0f}%, test: {test_size*100:.0f}%)...")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            test_size=test_size,
            random_state=random_state,
            stratify=self.y  # Garde la même proportion de fraudes
        )
        
        print(f"✅ Train: {len(self.X_train):,} samples")
        print(f"   - Fraudes: {self.y_train.sum():,} ({self.y_train.mean()*100:.3f}%)")
        print(f"✅ Test: {len(self.X_test):,} samples")
        print(f"   - Fraudes: {self.y_test.sum():,} ({self.y_test.mean()*100:.3f}%)")
        
        return self
    
    def scale_features(self):
        """
        Étape 4: Normaliser les features
        
        POURQUOI? Pour que toutes les features soient sur la même échelle
        """
        print("\n📏 Normalisation des features...")
        
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("✅ Features normalisées (mean=0, std=1)")
        
        return self
    
    def apply_smote(self, sampling_strategy=0.5):
        """
        Étape 5: Appliquer SMOTE pour gérer le class imbalance
        
        STRATÉGIE:
        - sampling_strategy=0.5 = créer des fraudes synthétiques jusqu'à 50% du nb de légitimes
        - Pas 100% car trop de synthétiques peut dégrader la qualité
        """
        print(f"\n🔄 Application de SMOTE (sampling_strategy={sampling_strategy})...")
        
        print(f"  Avant SMOTE:")
        print(f"    Légitimes: {(self.y_train == 0).sum():,}")
        print(f"    Fraudes: {(self.y_train == 1).sum():,}")
        
        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=42,
            k_neighbors=5
        )
        
        self.X_train_resampled, self.y_train_resampled = smote.fit_resample(
            self.X_train_scaled,
            self.y_train
        )
        
        print(f"  Après SMOTE:")
        print(f"    Légitimes: {(self.y_train_resampled == 0).sum():,}")
        print(f"    Fraudes: {(self.y_train_resampled == 1).sum():,}")
        
        fraud_increase = (self.y_train_resampled == 1).sum() / (self.y_train == 1).sum()
        print(f"✅ Fraudes augmentées de {fraud_increase:.1f}x (synthétiques)")
        
        return self
    
    def train_xgboost(self):
        """
        Étape 6: Entraîner XGBoost
        
        HYPERPARAMÈTRES OPTIMISÉS POUR LA FRAUDE:
        - scale_pos_weight: Pénalise plus les erreurs sur les fraudes
        - max_depth: Profondeur des arbres (6 = bon équilibre)
        - learning_rate: Vitesse d'apprentissage (0.1 = standard)
        - n_estimators: Nombre d'arbres (500 avec early stopping)
        - subsample: 80% des données par arbre (évite overfitting)
        """
        print("\n🎓 Entraînement du modèle XGBoost...")
        
        # Calculer le poids des classes
        n_legit = (self.y_train == 0).sum()
        n_fraud = (self.y_train == 1).sum()
        scale_pos_weight = n_legit / n_fraud
        
        print(f"  ⚖️  scale_pos_weight = {scale_pos_weight:.1f}")
        
        # Hyperparamètres
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 500,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': scale_pos_weight,
            'random_state': 42,
            'tree_method': 'hist',  # Plus rapide
            'n_jobs': -1  # Utiliser tous les CPU
        }
        
        # Créer le modèle
        self.model = xgb.XGBClassifier(**params)
        
        # Entraîner avec early stopping
        print("  🔄 Entraînement en cours...")
        start_time = time.time()
        
        self.model.fit(
            self.X_train_resampled,
            self.y_train_resampled,
            eval_set=[(self.X_test_scaled, self.y_test)],
            verbose=False  # Pas d'output verbeux
        )
        
        elapsed = time.time() - start_time
        print(f"✅ Modèle entraîné en {elapsed:.1f}s")
        
        # Sauvegarder les params pour MLflow
        self.params = params
        
        return self
    
    def evaluate_model(self):
        """
        Étape 7: Évaluer le modèle
        """
        print("\n📊 Évaluation du modèle...")
        
        # Prédictions
        y_pred_proba = self.model.predict_proba(self.X_test_scaled)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Métriques principales
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        avg_precision = average_precision_score(self.y_test, y_pred_proba)
        
        print(f"\n🎯 MÉTRIQUES PRINCIPALES:")
        print(f"  • ROC-AUC Score: {roc_auc:.4f}")
        print(f"  • Average Precision: {avg_precision:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        print(f"\n📈 CONFUSION MATRIX:")
        print(f"  True Negatives (TN):  {tn:,}")
        print(f"  False Positives (FP): {fp:,} ⚠️  (Fausses alertes)")
        print(f"  False Negatives (FN): {fn:,} 🚨 (Fraudes manquées)")
        print(f"  True Positives (TP):  {tp:,} ✅ (Fraudes détectées)")
        
        # Métriques dérivées
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n💯 MÉTRIQUES DÉTAILLÉES:")
        print(f"  • Precision: {precision:.4f} (Fiabilité des alertes)")
        print(f"  • Recall: {recall:.4f} (% de fraudes détectées)")
        print(f"  • F1-Score: {f1:.4f}")
        
        # Sauvegarder les métriques
        self.metrics = {
            'roc_auc': roc_auc,
            'avg_precision': avg_precision,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        }
        
        return self
    
    def log_to_mlflow(self):
        """
        Étape 8: Logger dans MLflow (version simplifiée)
        """
        print("\n📝 Logging dans MLflow...")
        
        try:
            with mlflow.start_run(run_name="xgboost_fraud_detector"):
                # Logger les paramètres
                mlflow.log_params(self.params)
                
                # Logger les métriques
                mlflow.log_metrics(self.metrics)
                
                # Logger les artifacts (features, config)
                mlflow.log_param("features", ", ".join(self.feature_names))
                mlflow.log_param("smote_applied", True)
                mlflow.log_param("scaler", "StandardScaler")
                mlflow.log_param("n_features", len(self.feature_names))
                
                # Sauvegarder le modèle comme artifact (pas avec log_model)
                import tempfile
                import shutil
                
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_model_path = Path(tmp_dir) / "model.pkl"
                    with open(tmp_model_path, 'wb') as f:
                        pickle.dump(self.model, f)
                    mlflow.log_artifact(str(tmp_model_path), "model")
                
                print("✅ Run enregistrée dans MLflow!")
                print(f"   📁 MLflow UI: mlflow ui --backend-store-uri file:///{MLFLOW_DIR}")
        
        except Exception as e:
            print(f"⚠️  Warning: MLflow logging failed: {e}")
            print("   (Le modèle est sauvegardé localement, c'est juste MLflow qui a un problème)")
        
        return self
    
    def save_model(self):
        """
        Étape 9: Sauvegarder le modèle et les artifacts
        """
        print("\n💾 Sauvegarde du modèle...")
        
        # Sauvegarder le modèle
        model_path = MODELS_DIR / "fraud_detector_v1.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"✅ Modèle: {model_path}")
        
        # Sauvegarder le scaler
        scaler_path = MODELS_DIR / "scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"✅ Scaler: {scaler_path}")
        
        # Sauvegarder les label encoders
        encoders_path = MODELS_DIR / "label_encoders.pkl"
        with open(encoders_path, 'wb') as f:
            pickle.dump(self.label_encoders, f)
        print(f"✅ Encoders: {encoders_path}")
        
        # Sauvegarder les noms de features
        features_path = MODELS_DIR / "feature_names.pkl"
        with open(features_path, 'wb') as f:
            pickle.dump(self.feature_names, f)
        print(f"✅ Feature names: {features_path}")
        
        # Sauvegarder les métriques
        metrics_path = MODELS_DIR / "metrics.pkl"
        with open(metrics_path, 'wb') as f:
            pickle.dump(self.metrics, f)
        print(f"✅ Metrics: {metrics_path}")
        
        return self
    
    def plot_feature_importance(self):
        """
        Étape 10: Visualiser l'importance des features
        """
        print("\n📊 Génération du graphique d'importance des features...")
        
        # Obtenir l'importance
        importance = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Top 15 features
        top_features = feature_importance_df.head(15)
        
        # Plot
        plt.figure(figsize=(12, 8))
        sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
        plt.title('Top 15 Most Important Features for Fraud Detection', fontsize=16, fontweight='bold')
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        
        # Sauvegarder
        plot_path = MODELS_DIR / "feature_importance.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Graphique sauvegardé: {plot_path}")
        
        print("\n🏆 TOP 10 FEATURES:")
        for i, row in top_features.head(10).iterrows():
            print(f"  {i+1:2d}. {row['feature']:25s} → {row['importance']:.4f}")
        
        return self
    
    def run_full_pipeline(self):
        """
        Exécuter le pipeline complet
        """
        start_time = time.time()
        
        (self
         .load_data()
         .prepare_features()
         .split_data()
         .scale_features()
         .apply_smote()
         .train_xgboost()
         .evaluate_model()
         .log_to_mlflow()
         .save_model()
         .plot_feature_importance())
        
        elapsed = time.time() - start_time
        
        print("\n" + "="*70)
        print(f"✅ PIPELINE TERMINÉ EN {elapsed/60:.1f} MINUTES!")
        print("="*70)
        
        print("\n🎉 RÉCAPITULATIF:")
        print(f"  • ROC-AUC: {self.metrics['roc_auc']:.4f}")
        print(f"  • Recall: {self.metrics['recall']:.4f}")
        print(f"  • Precision: {self.metrics['precision']:.4f}")
        print(f"  • Fraudes détectées: {self.metrics['true_positives']}/{self.metrics['true_positives'] + self.metrics['false_negatives']}")
        
        print("\n📁 FICHIERS CRÉÉS:")
        print(f"  • Modèle: {MODELS_DIR}/fraud_detector_v1.pkl")
        print(f"  • Importance: {MODELS_DIR}/feature_importance.png")
        
        print("\n🔍 PROCHAINES ÉTAPES:")
        print("  1. Visualiser MLflow UI:")
        print(f"     → cd {BASE_DIR}")
        print(f"     → mlflow ui --backend-store-uri file:///{MLFLOW_DIR}")
        print("  2. Tester le modèle:")
        print("     → python src/ml_models/predict_fraud.py")


def main():
    """Point d'entrée"""
    trainer = FraudModelTrainer()
    trainer.run_full_pipeline()


if __name__ == "__main__":
    main()