"""
Feature Engineering & Synthetic Entity Generation
================================================
Ce script crée des entités synthétiques pour construire un Knowledge Graph:
- Customers (clients)
- Accounts (comptes bancaires)
- Devices (téléphones, ordinateurs)
- Locations (villes, pays)

Il génère aussi des patterns de fraude réalistes.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import hashlib

# Configuration des chemins
BASE_DIR = Path(__file__).parent.parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DATA_DIR.mkdir(exist_ok=True)

class FraudFeatureEngineer:    
    def __init__(self, train_path: str, test_path: str = None):
        """
        Initialise l'engineer
        
        Args:
            train_path: Chemin vers fraudTrain.csv
            test_path: Chemin vers fraudTest.csv (optionnel)
        """
        print("📊 Chargement des données...")
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path) if test_path else None
        
        # Combiner train et test pour l'analyse
        self.df = self.train_df.copy()
        if self.test_df is not None:
            self.df = pd.concat([self.train_df, self.test_df], ignore_index=True)
        
        print(f"✅ {len(self.df)} transactions chargées")
        
        # Convertir les dates
        self.df['trans_date_trans_time'] = pd.to_datetime(
            self.df['trans_date_trans_time']
        )
        self.df['dob'] = pd.to_datetime(self.df['dob'])
        
    def create_customer_ids(self) -> pd.DataFrame:
        """
        Étape 1: Créer des IDs uniques pour les customers
        
        LOGIQUE:
        - Chaque combinaison (first, last, dob) = 1 customer unique
        - On génère un customer_id stable avec hash
        """
        print("\n🔑 Génération des Customer IDs...")
        
        # Créer un identifiant unique basé sur first + last + dob
        self.df['customer_id'] = self.df.apply(
            lambda row: hashlib.sha256(
                f"{row['first']}_{row['last']}_{row['dob']}".encode()
            ).hexdigest()[:16],
            axis=1
        )
        
        n_customers = self.df['customer_id'].nunique()
        print(f"✅ {n_customers} customers uniques créés")
        
        return self.df
    
    def create_account_ids(self) -> pd.DataFrame:
        """
        Étape 2: Créer des Account IDs
        
        LOGIQUE:
        - On utilise 'cc_num' (numéro de carte) comme base
        - Chaque carte = 1 compte bancaire
        - Un customer peut avoir plusieurs accounts
        """
        print("\n💳 Génération des Account IDs...")
        
        # Utiliser cc_num comme base pour l'account
        self.df['account_id'] = 'ACC_' + self.df['cc_num'].astype(str)
        
        n_accounts = self.df['account_id'].nunique()
        print(f"✅ {n_accounts} accounts créés")
        
        # Statistique: combien de comptes par customer?
        accounts_per_customer = self.df.groupby('customer_id')['account_id'].nunique()
        print(f"📊 Moyenne: {accounts_per_customer.mean():.2f} comptes/customer")
        
        return self.df
    
    def create_device_ids(self) -> pd.DataFrame:
        """
        Étape 3: Créer des Device IDs (CRITICAL POUR FRAUDE!)
        
        LOGIQUE:
        - Les fraudeurs utilisent souvent le MÊME device pour plusieurs comptes
        - On crée des patterns réalistes:
          * Frauds: 30% partagent des devices
          * Légitimes: 5% partagent (famille)
        """
        print("\n📱 Génération des Device IDs...")
        
        np.random.seed(42)
        
        # Initialiser la colonne
        self.df['device_id'] = None
        
        # FRAUD TRANSACTIONS: Créer des clusters de devices partagés
        fraud_mask = self.df['is_fraud'] == 1
        fraud_indices = self.df[fraud_mask].index
        
        n_fraud = len(fraud_indices)
        print(f"🔴 {n_fraud} transactions frauduleuses à traiter...")
        
        # 30% des frauds partagent des devices (pattern de fraude!)
        n_shared_fraud_devices = int(n_fraud * 0.3)
        n_unique_fraud_devices = n_fraud - n_shared_fraud_devices
        
        # Devices partagés (2-5 transactions par device)
        shared_fraud_devices = []
        remaining = n_shared_fraud_devices
        device_counter = 0
        
        while remaining > 0:
            n_transactions = np.random.randint(2, 6)  # 2 à 5 tx par device
            n_transactions = min(n_transactions, remaining)
            device_id = f"DEV_FRAUD_{device_counter:06d}"
            shared_fraud_devices.extend([device_id] * n_transactions)
            remaining -= n_transactions
            device_counter += 1
        
        # Devices uniques
        unique_fraud_devices = [
            f"DEV_FRAUD_UNIQUE_{i:06d}" 
            for i in range(n_unique_fraud_devices)
        ]
        
        # Combiner et assigner
        all_fraud_devices = shared_fraud_devices + unique_fraud_devices
        np.random.shuffle(all_fraud_devices)
        self.df.loc[fraud_indices, 'device_id'] = all_fraud_devices[:n_fraud]
        
        # LEGITIMATE TRANSACTIONS: Devices majoritairement uniques
        legit_mask = self.df['is_fraud'] == 0
        legit_indices = self.df[legit_mask].index
        
        n_legit = len(legit_indices)
        print(f"🟢 {n_legit} transactions légitimes à traiter...")
        
        # 5% partagent des devices (famille, conjoint)
        n_shared_legit_devices = int(n_legit * 0.05)
        n_unique_legit_devices = n_legit - n_shared_legit_devices
        
        # Devices partagés légitimes (2-3 personnes par device)
        shared_legit_devices = []
        remaining = n_shared_legit_devices
        device_counter = 0
        
        while remaining > 0:
            n_transactions = np.random.randint(2, 4)
            n_transactions = min(n_transactions, remaining)
            device_id = f"DEV_LEGIT_SHARED_{device_counter:06d}"
            shared_legit_devices.extend([device_id] * n_transactions)
            remaining -= n_transactions
            device_counter += 1
        
        # Devices uniques légitimes
        unique_legit_devices = [
            f"DEV_LEGIT_{i:06d}" 
            for i in range(n_unique_legit_devices)
        ]
        
        all_legit_devices = shared_legit_devices + unique_legit_devices
        np.random.shuffle(all_legit_devices)
        self.df.loc[legit_indices, 'device_id'] = all_legit_devices[:n_legit]
        
        # Statistiques
        n_devices = self.df['device_id'].nunique()
        print(f"✅ {n_devices} devices créés")
        
        # Trouver les devices suspects (utilisés par multiples accounts)
        device_account_counts = self.df.groupby('device_id')['account_id'].nunique()
        suspicious_devices = device_account_counts[device_account_counts > 1]
        print(f"⚠️  {len(suspicious_devices)} devices suspects (>1 account)")
        
        return self.df
    
    def create_location_ids(self) -> pd.DataFrame:
        """
        Étape 4: Créer des Location IDs
        
        LOGIQUE:
        - Combiner city + state = location unique
        - Ajouter des coordonnées lat/long synthétiques
        """
        print("\n📍 Génération des Location IDs...")
        
        # Créer location_id
        self.df['location_id'] = (
            'LOC_' + 
            self.df['city'].str.upper().str.replace(' ', '_') + 
            '_' + 
            self.df['state']
        )
        
        n_locations = self.df['location_id'].nunique()
        print(f"✅ {n_locations} locations créées")
        
        return self.df
    
    def engineer_fraud_features(self) -> pd.DataFrame:
        """
        Étape 5: Créer des FEATURES de FRAUDE avancées
        
        FEATURES CRÉÉES:
        1. Transaction Velocity (tx par heure)
        2. Amount Deviation (écart vs moyenne user)
        3. Night Transaction (activité nocturne)
        4. Weekend Transaction
        5. High-Risk Merchant Category
        6. Customer Age
        7. Days Since First Transaction
        """
        print("\n🧮 Engineering des fraud features...")
        
        # Feature 1: HEURE et JOUR de la semaine
        self.df['hour'] = self.df['trans_date_trans_time'].dt.hour
        self.df['day_of_week'] = self.df['trans_date_trans_time'].dt.dayofweek
        self.df['is_weekend'] = (self.df['day_of_week'] >= 5).astype(int)
        self.df['is_night'] = ((self.df['hour'] >= 22) | (self.df['hour'] <= 6)).astype(int)
        
        print("✅ Features temporelles créées")
        
        # Feature 2: ÂGE du customer
        self.df['customer_age'] = (
            self.df['trans_date_trans_time'] - self.df['dob']
        ).dt.days / 365.25
        
        print("✅ Customer age calculé")
        
        # Feature 3: VELOCITY (transactions par customer par heure)
        self.df['trans_hour_bucket'] = self.df['trans_date_trans_time'].dt.floor('h')
        
        velocity = self.df.groupby(['customer_id', 'trans_hour_bucket']).size()
        velocity_dict = velocity.to_dict()
        
        self.df['tx_velocity'] = self.df.apply(
            lambda row: velocity_dict.get((row['customer_id'], row['trans_hour_bucket']), 1),
            axis=1
        )
        
        print("✅ Transaction velocity calculée")
        
        # Feature 4: AMOUNT DEVIATION (écart vs moyenne du customer)
        customer_avg_amt = self.df.groupby('customer_id')['amt'].transform('mean')
        customer_std_amt = self.df.groupby('customer_id')['amt'].transform('std')
        
        self.df['amt_deviation'] = (
            (self.df['amt'] - customer_avg_amt) / (customer_std_amt + 1)
        )
        
        print("✅ Amount deviation calculée")
        
        # Feature 5: HIGH-RISK CATEGORIES
        high_risk_categories = ['gas_transport', 'shopping_net', 'misc_net', 'grocery_pos']
        self.df['is_high_risk_category'] = (
            self.df['category'].isin(high_risk_categories)
        ).astype(int)
        
        print("✅ High-risk categories identifiées")
        
        # Feature 6: TRANSACTION COUNT par customer (historique)
        self.df = self.df.sort_values(['customer_id', 'trans_date_trans_time'])
        self.df['customer_tx_count'] = self.df.groupby('customer_id').cumcount() + 1
        
        print("✅ Transaction count calculé")
        
        # Feature 7: DAYS SINCE FIRST TRANSACTION
        first_tx_date = self.df.groupby('customer_id')['trans_date_trans_time'].transform('min')
        self.df['days_since_first_tx'] = (
            self.df['trans_date_trans_time'] - first_tx_date
        ).dt.days
        
        print("✅ Days since first tx calculé")
        
        # Statistiques finales
        print("\n📊 STATISTIQUES DES FEATURES:")
        print(f"   - Transaction velocity: {self.df['tx_velocity'].mean():.2f} (avg)")
        print(f"   - Night transactions: {self.df['is_night'].sum()} ({self.df['is_night'].mean()*100:.1f}%)")
        print(f"   - Weekend transactions: {self.df['is_weekend'].sum()} ({self.df['is_weekend'].mean()*100:.1f}%)")
        print(f"   - High-risk category: {self.df['is_high_risk_category'].sum()} ({self.df['is_high_risk_category'].mean()*100:.1f}%)")
        
        return self.df
    
    def save_processed_data(self) -> None:
        """
        Étape 6: Sauvegarder les données enrichies
        """
        print("\n💾 Sauvegarde des données enrichies...")
        
        # Fichier principal
        output_path = PROCESSED_DATA_DIR / "fraud_transactions_enriched.csv"
        self.df.to_csv(output_path, index=False)
        print(f"✅ Sauvegardé: {output_path}")
        
        # Créer des tables séparées pour Neo4j
        self._save_neo4j_tables()
        
    def _save_neo4j_tables(self) -> None:
        """
        Créer des tables CSV séparées pour chaque type de nœud Neo4j
        """
        print("\n📦 Création des tables Neo4j...")
        
        # Table CUSTOMERS
        customers = self.df[[
            'customer_id', 'first', 'last', 'gender', 
            'street', 'city', 'state', 'zip', 'dob', 'customer_age'
        ]].drop_duplicates(subset=['customer_id'])
        
        customers_path = PROCESSED_DATA_DIR / "neo4j_customers.csv"
        customers.to_csv(customers_path, index=False)
        print(f"✅ {len(customers)} customers → {customers_path.name}")
        
        # Table ACCOUNTS
        accounts = self.df[[
            'account_id', 'customer_id', 'cc_num'
        ]].drop_duplicates(subset=['account_id'])
        
        accounts_path = PROCESSED_DATA_DIR / "neo4j_accounts.csv"
        accounts.to_csv(accounts_path, index=False)
        print(f"✅ {len(accounts)} accounts → {accounts_path.name}")
        
        # Table MERCHANTS
        merchants = self.df[[
            'merchant', 'category'
        ]].drop_duplicates(subset=['merchant'])
        
        merchants['merchant_id'] = 'MERCH_' + merchants.index.astype(str)
        merchants_path = PROCESSED_DATA_DIR / "neo4j_merchants.csv"
        merchants.to_csv(merchants_path, index=False)
        print(f"✅ {len(merchants)} merchants → {merchants_path.name}")
        
        # Table DEVICES
        devices = self.df[['device_id']].drop_duplicates()
        devices_path = PROCESSED_DATA_DIR / "neo4j_devices.csv"
        devices.to_csv(devices_path, index=False)
        print(f"✅ {len(devices)} devices → {devices_path.name}")
        
        # Table LOCATIONS
        locations = self.df[[
            'location_id', 'city', 'state', 'lat', 'long'
        ]].drop_duplicates(subset=['location_id'])
        
        locations_path = PROCESSED_DATA_DIR / "neo4j_locations.csv"
        locations.to_csv(locations_path, index=False)
        print(f"✅ {len(locations)} locations → {locations_path.name}")
        
        # Table TRANSACTIONS (avec toutes les relations)
        transactions = self.df[[
            'trans_num', 'trans_date_trans_time', 'account_id', 
            'merchant', 'device_id', 'location_id',
            'amt', 'is_fraud', 'category',
            'hour', 'day_of_week', 'is_weekend', 'is_night',
            'tx_velocity', 'amt_deviation', 'is_high_risk_category',
            'customer_tx_count', 'days_since_first_tx'
        ]].copy()
        
        transactions['transaction_id'] = 'TX_' + transactions['trans_num']
        transactions_path = PROCESSED_DATA_DIR / "neo4j_transactions.csv"
        transactions.to_csv(transactions_path, index=False)
        print(f"✅ {len(transactions)} transactions → {transactions_path.name}")
        
        print("\n🎉 Toutes les tables Neo4j sont prêtes!")


def main():
    """
    Point d'entrée principal
    """
    print("="*60)
    print("🚀 FRAUD DETECTION - FEATURE ENGINEERING")
    print("="*60)
    
    # Chemins des données
    train_path = RAW_DATA_DIR / "fraudTrain.csv"
    test_path = RAW_DATA_DIR / "fraudTest.csv"
    
    # Initialiser l'engineer
    engineer = FraudFeatureEngineer(
        train_path=str(train_path),
        test_path=str(test_path)
    )
    
    # Pipeline complet
    engineer.create_customer_ids()
    engineer.create_account_ids()
    engineer.create_device_ids()
    engineer.create_location_ids()
    engineer.engineer_fraud_features()
    engineer.save_processed_data()
    
    print("\n" + "="*60)
    print("✅ FEATURE ENGINEERING TERMINÉ!")
    print("="*60)
    print(f"\n📁 Fichiers créés dans: {PROCESSED_DATA_DIR}")
    print("\n🎯 PROCHAINE ÉTAPE: Installation de Neo4j")


if __name__ == "__main__":
    main()