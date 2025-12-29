"""
Fraud Prediction Script
=======================
Teste le modèle entraîné sur des transactions spécifiques
"""

import pandas as pd
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"


class FraudPredictor:
    """Classe pour faire des prédictions de fraude"""
    
    def __init__(self):
        """Charger le modèle et les artifacts"""
        print("🔮 Fraud Predictor")
        print("="*50)
        
        # Charger le modèle
        print("📦 Chargement du modèle...")
        with open(MODELS_DIR / "fraud_detector_v1.pkl", 'rb') as f:
            self.model = pickle.load(f)
        
        # Charger le scaler
        with open(MODELS_DIR / "scaler.pkl", 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Charger les encoders
        with open(MODELS_DIR / "label_encoders.pkl", 'rb') as f:
            self.label_encoders = pickle.load(f)
        
        # Charger les feature names
        with open(MODELS_DIR / "feature_names.pkl", 'rb') as f:
            self.feature_names = pickle.load(f)
        
        print("✅ Modèle chargé!")
    
    def predict_single(self, transaction: dict) -> dict:
        """
        Prédire si une transaction est frauduleuse
        
        Args:
            transaction: Dict avec les features de la transaction
        
        Returns:
            Dict avec la prédiction et le score
        """
        # Créer un DataFrame
        df = pd.DataFrame([transaction])
        
        # Encoder les catégorielles
        for col in ['category', 'gender', 'state']:
            if col in df.columns and col in self.label_encoders:
                df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        # S'assurer d'avoir toutes les features
        for feat in self.feature_names:
            if feat not in df.columns:
                df[feat] = 0
        
        # Réordonner les colonnes
        df = df[self.feature_names]
        
        # Scaler
        X_scaled = self.scaler.transform(df)
        
        # Prédire
        fraud_proba = self.model.predict_proba(X_scaled)[0, 1]
        is_fraud = fraud_proba >= 0.5
        
        # Risk level
        if fraud_proba < 0.3:
            risk_level = "LOW"
        elif fraud_proba < 0.7:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"
        
        return {
            'is_fraud': bool(is_fraud),
            'fraud_probability': float(fraud_proba),
            'risk_level': risk_level
        }
    
    def test_on_real_data(self, n_samples=10):
        """Tester sur des vraies transactions"""
        print(f"\n🧪 Test sur {n_samples} transactions réelles...")
        
        # Charger les données
        df = pd.read_csv(DATA_DIR / "fraud_transactions_enriched.csv")
        
        # Prendre quelques fraudes et quelques légitimes
        fraud_samples = df[df['is_fraud'] == 1].sample(n=n_samples//2, random_state=42)
        legit_samples = df[df['is_fraud'] == 0].sample(n=n_samples//2, random_state=42)
        
        samples = pd.concat([fraud_samples, legit_samples])
        
        print("\n" + "="*80)
        
        correct = 0
        for idx, row in samples.iterrows():
            # Préparer la transaction
            transaction = {feat: row[feat] for feat in self.feature_names if feat in row}
            
            # Prédire
            result = self.predict_single(transaction)
            
            # Vérifier
            actual = bool(row['is_fraud'])
            predicted = result['is_fraud']
            is_correct = actual == predicted
            
            if is_correct:
                correct += 1
            
            # Afficher
            status = "✅" if is_correct else "❌"
            print(f"{status} Transaction #{row['trans_num']}")
            print(f"   Actual: {'FRAUD' if actual else 'LEGIT'} | "
                  f"Predicted: {'FRAUD' if predicted else 'LEGIT'} | "
                  f"Probability: {result['fraud_probability']:.2%} | "
                  f"Risk: {result['risk_level']}")
            print(f"   Amount: ${row['amt']:.2f} | Hour: {row['hour']}h | "
                  f"Velocity: {row['tx_velocity']:.1f}")
            print()
        
        accuracy = correct / n_samples
        print("="*80)
        print(f"🎯 Accuracy sur l'échantillon: {accuracy:.1%} ({correct}/{n_samples})")


def main():
    """Point d'entrée"""
    predictor = FraudPredictor()
    
    # Test sur des données réelles
    predictor.test_on_real_data(n_samples=20)
    
    print("\n💡 EXEMPLE D'UTILISATION:")
    print("""
    from predict_fraud import FraudPredictor
    
    predictor = FraudPredictor()
    
    transaction = {
        'amt': 5000.0,
        'hour': 3,
        'day_of_week': 2,
        'is_weekend': 0,
        'is_night': 1,
        'tx_velocity': 8,
        'amt_deviation': 15.2,
        'customer_tx_count': 234,
        'days_since_first_tx': 450,
        'is_high_risk_category': 1,
        'customer_age': 35,
        'category': 'shopping_net',
        'gender': 'M',
        'state': 'CA'
    }
    
    result = predictor.predict_single(transaction)
    print(result)
    # {'is_fraud': True, 'fraud_probability': 0.89, 'risk_level': 'HIGH'}
    """)


if __name__ == "__main__":
    main()