"""
Fraud Detection GraphRAG System - VERSION CORRIGÉE
===================================================
Système d'explication de fraude utilisant:
- Neo4j Knowledge Graph (driver natif)
- XGBoost ML Model
- Ollama LLM (Llama 3.2)
"""

from neo4j import GraphDatabase
import pickle
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

# LangChain (seulement pour le LLM)
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import warnings
import logging

# Supprimer les warnings Neo4j
warnings.filterwarnings('ignore')
logging.getLogger('neo4j').setLevel(logging.ERROR)

# Configuration
BASE_DIR = Path(__file__).parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"


class FraudGraphRAG:
    """Système GraphRAG pour l'explication de fraudes"""
    
    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "fraudpassword",
        llm_model: str = "llama3.2"
    ):
        """Initialiser le système GraphRAG"""
        print("="*70)
        print("🚀 FRAUD GRAPHRAG SYSTEM - INITIALISATION")
        print("="*70)
        
        # 1. Connexion Neo4j avec driver natif
        print("\n🔌 Connexion au Knowledge Graph...")
        self.driver = GraphDatabase.driver(
            neo4j_uri,
            auth=(neo4j_user, neo4j_password)
        )
        
        # Test de connexion
        with self.driver.session() as session:
            result = session.run("RETURN 1 AS test")
            result.single()
        
        print("✅ Connecté à Neo4j")
        
        # 2. Charger le LLM
        print(f"\n🤖 Chargement du LLM ({llm_model})...")
        self.llm = ChatOllama(
            model=llm_model,
            temperature=0.3,
            num_predict=512
        )
        print("✅ LLM chargé")
        
        # 3. Charger le modèle ML
        print("\n📊 Chargement du modèle ML...")
        with open(MODELS_DIR / "fraud_detector_v1.pkl", 'rb') as f:
            self.ml_model = pickle.load(f)
        
        with open(MODELS_DIR / "scaler.pkl", 'rb') as f:
            self.scaler = pickle.load(f)
        
        with open(MODELS_DIR / "label_encoders.pkl", 'rb') as f:
            self.label_encoders = pickle.load(f)
        
        with open(MODELS_DIR / "feature_names.pkl", 'rb') as f:
            self.feature_names = pickle.load(f)
        
        print("✅ Modèle ML chargé")
        
        print("\n" + "="*70)
        print("✅ GRAPHRAG PRÊT!")
        print("="*70)
    
    def close(self):
        """Fermer la connexion Neo4j"""
        if self.driver:
            self.driver.close()
    
    def get_transaction_details(self, tx_id: str) -> Optional[Dict]:
        """Récupérer les détails d'une transaction depuis le graphe"""
        query = """
        MATCH (t:Transaction {id: $tx_id})
        OPTIONAL MATCH (t)<-[:MADE]-(a:Account)<-[:OWNS]-(c:Customer)
        OPTIONAL MATCH (t)-[:USING]->(d:Device)
        OPTIONAL MATCH (t)-[:AT]->(m:Merchant)
        OPTIONAL MATCH (t)-[:IN]->(l:Location)
        
        RETURN 
            t.id as tx_id,
            t.trans_num as trans_num,
            t.amount as amount,
            t.timestamp as timestamp,
            t.is_fraud as is_fraud,
            t.category as category,
            t.hour as hour,
            t.is_night as is_night,
            t.is_weekend as is_weekend,
            t.tx_velocity as velocity,
            t.amt_deviation as amt_deviation,
            t.customer_tx_count as customer_tx_count,
            c.first_name as customer_first,
            c.last_name as customer_last,
            c.age as customer_age,
            d.id as device_id,
            m.name as merchant_name,
            m.category as merchant_category,
            l.city as location_city,
            l.state as location_state
        """
        
        with self.driver.session() as session:
            result = session.run(query, tx_id=tx_id)
            records = [dict(record) for record in result]
            
            if not records:
                return None
            
            return records[0]
    
    def get_device_network(self, device_id: str) -> Dict:
        """Analyser le réseau de comptes utilisant un device"""
        
        # Première query: statistiques globales
        stats_query = """
        MATCH (d:Device {id: $device_id})<-[:USING]-(t:Transaction)
              <-[:MADE]-(a:Account)
        
        RETURN 
            COUNT(DISTINCT a) as account_count,
            COUNT(t) as tx_count,
            SUM(CASE WHEN t.is_fraud = 1 THEN 1 ELSE 0 END) as fraud_count
        """
        
        # Deuxième query: détails des comptes
        accounts_query = """
        MATCH (d:Device {id: $device_id})<-[:USING]-(t:Transaction)
              <-[:MADE]-(a:Account)<-[:OWNS]-(c:Customer)
        
        WITH a, c,
             COUNT(t) as tx_count,
             SUM(CASE WHEN t.is_fraud = 1 THEN 1 ELSE 0 END) as fraud_count
        
        RETURN 
            c.first_name + ' ' + c.last_name as customer,
            a.id as account,
            tx_count,
            fraud_count
        ORDER BY fraud_count DESC
        LIMIT 10
        """
        
        with self.driver.session() as session:
            # Récupérer les stats
            result = session.run(stats_query, device_id=device_id)
            stats = [dict(record) for record in result]
            
            if not stats:
                return {
                    "account_count": 0,
                    "tx_count": 0,
                    "fraud_count": 0,
                    "connected_accounts": []
                }
            
            # Récupérer les comptes
            result = session.run(accounts_query, device_id=device_id)
            accounts = [dict(record) for record in result]
            
            return {
                "account_count": stats[0]['account_count'],
                "tx_count": stats[0]['tx_count'],
                "fraud_count": stats[0]['fraud_count'],
                "connected_accounts": accounts
            }
    
    def predict_transaction(self, tx_details: Dict) -> Dict:
        """Faire une prédiction ML sur la transaction"""
        # Préparer les features
        features = {}
        
        # Features numériques
        numeric_features = [
            'amount', 'hour', 'is_weekend', 'is_night',
            'velocity', 'amt_deviation', 'customer_tx_count',
            'customer_age'
        ]
        
        for feat in numeric_features:
            features[feat] = tx_details.get(feat, 0)
        
        # Features additionnelles manquantes
        features['day_of_week'] = 0
        features['is_high_risk_category'] = 0
        features['days_since_first_tx'] = 0
        
        # Features catégorielles
        features['category'] = tx_details.get('category', 'unknown')
        features['gender'] = 'M'
        features['state'] = tx_details.get('location_state', 'CA')
        
        # Créer DataFrame
        df = pd.DataFrame([features])
        
        # Encoder catégorielles
        for col in ['category', 'gender', 'state']:
            if col in self.label_encoders:
                try:
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
                except:
                    df[col] = 0
        
        # Assurer l'ordre des features
        df = df.reindex(columns=self.feature_names, fill_value=0)
        
        # Scaler
        X_scaled = self.scaler.transform(df)
        
        # Prédire
        fraud_proba = self.ml_model.predict_proba(X_scaled)[0, 1]
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
    
    def explain_transaction(self, tx_id: str) -> str:
        """Générer une explication complète pour une transaction"""
        print(f"\n🔍 Analyse de la transaction {tx_id}...")
        
        # 1. Récupérer les détails
        tx_details = self.get_transaction_details(tx_id)
        
        if not tx_details:
            return f"❌ Transaction {tx_id} introuvable dans le graphe."
        
        # 2. Prédiction ML
        ml_prediction = self.predict_transaction(tx_details)
        
        # 3. Analyse du réseau device
        device_network = self.get_device_network(tx_details['device_id'])
        
        # 4. Helper pour gérer les valeurs None
        def safe_value(value, default=0):
            return value if value is not None else default
        
        def safe_bool(value):
            return 'Oui' if value else 'Non'
        
        # 5. Extraire timestamp si disponible
        timestamp = tx_details.get('timestamp')
        hour_info = f"{safe_value(tx_details.get('hour'))}h" if tx_details.get('hour') is not None else "N/A"
        
        # 6. Construire le contexte avec gestion des None
        context = f"""
DÉTAILS DE LA TRANSACTION:
- ID: {tx_id}
- Montant: ${safe_value(tx_details.get('amount')):.2f}
- Date/Heure: {timestamp if timestamp else 'N/A'}
- Heure: {hour_info}
- Transaction nocturne: {safe_bool(tx_details.get('is_night'))}
- Weekend: {safe_bool(tx_details.get('is_weekend'))}
- Client: {tx_details.get('customer_first', 'N/A')} {tx_details.get('customer_last', 'N/A')} ({safe_value(tx_details.get('customer_age'))} ans)
- Marchand: {tx_details.get('merchant_name', 'N/A')} ({tx_details.get('merchant_category', 'N/A')})
- Localisation: {tx_details.get('location_city', 'N/A')}, {tx_details.get('location_state', 'N/A')}
- Catégorie: {tx_details.get('category', 'N/A')}

PRÉDICTION ML:
- Fraude: {'OUI' if ml_prediction['is_fraud'] else 'NON'}
- Probabilité: {ml_prediction['fraud_probability']:.2%}
- Niveau de risque: {ml_prediction['risk_level']}

ANALYSE DU DEVICE ({tx_details.get('device_id', 'N/A')}):
- Nombre de comptes utilisant ce device: {device_network['account_count']}
- Nombre total de transactions: {device_network['tx_count']}
- Transactions frauduleuses: {device_network['fraud_count']} ({device_network['fraud_count']/max(device_network['tx_count'], 1)*100:.1f}%)

PATTERNS COMPORTEMENTAUX:
- Vélocité de transaction: {safe_value(tx_details.get('velocity'))} tx/heure
- Déviation du montant: {safe_value(tx_details.get('amt_deviation')):.2f}x de la normale
- Historique du client: {safe_value(tx_details.get('customer_tx_count'))} transactions
"""
        
        # 5. Prompt pour le LLM
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Tu es un expert en détection de fraude bancaire.

Ton rôle est d'analyser les transactions et d'expliquer clairement:
1. Si la transaction est frauduleuse ou non
2. Les raisons principales (device suspect, montant anormal, comportement inhabituel)
3. Les preuves du graphe de connaissances
4. Une recommandation d'action

Sois précis, professionnel et structure ta réponse avec des sections claires.
Utilise des émojis pour la lisibilité."""),
            ("user", """Analyse cette transaction et fournis une explication détaillée:

{context}

Structure ta réponse ainsi:
1. 📊 VERDICT
2. 🔍 ANALYSE DÉTAILLÉE
3. ⚠️ SIGNAUX D'ALERTE (ou ✅ SIGNAUX POSITIFS)
4. 💡 RECOMMANDATION""")
        ])
        
        # 6. Générer l'explication
        chain = prompt | self.llm | StrOutputParser()
        
        print("🤖 Génération de l'explication par le LLM...")
        explanation = chain.invoke({"context": context})
        
        return explanation
    
    def investigate_fraud_network(self, device_id: str) -> str:
        """Investiguer un réseau de fraude autour d'un device"""
        print(f"\n🕵️ Investigation du réseau pour device {device_id}...")
        
        # Récupérer les données du réseau
        network = self.get_device_network(device_id)
        
        if network['account_count'] == 0:
            return f"❌ Aucune transaction trouvée pour le device {device_id}"
        
        # Construire le contexte
        context = f"""
DEVICE ID: {device_id}

STATISTIQUES DU RÉSEAU:
- Nombre de comptes connectés: {network['account_count']}
- Nombre total de transactions: {network['tx_count']}
- Transactions frauduleuses: {network['fraud_count']} ({network['fraud_count']/max(network['tx_count'], 1)*100:.1f}%)

COMPTES CONNECTÉS:
"""
        
        # Ajouter les détails des comptes
        for i, acc in enumerate(network['connected_accounts'][:10], 1):
            fraud_rate = (acc['fraud_count'] / max(acc['tx_count'], 1) * 100)
            context += f"\n{i}. {acc['customer']} (Compte: {acc['account']})"
            context += f"\n   - Transactions: {acc['tx_count']}"
            context += f"\n   - Fraudes: {acc['fraud_count']} ({fraud_rate:.1f}%)"
        
        # Prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Tu es un expert en investigation de réseaux de fraude.

Analyse les patterns et identifie:
1. Le type de réseau (mule network, bot, fraude organisée, etc.)
2. Le niveau de menace
3. Les recommandations d'action

Sois précis et factuel."""),
            ("user", """Analyse ce réseau et fournis un rapport d'investigation:

{context}

Structure ta réponse:
1. 🕸️ TYPE DE RÉSEAU
2. 🚨 NIVEAU DE MENACE
3. 📊 ANALYSE DES PATTERNS
4. 🎯 RECOMMANDATIONS""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        print("🤖 Génération du rapport d'investigation...")
        report = chain.invoke({"context": context})
        
        return report
    
    def find_suspicious_devices(self, min_accounts: int = 3) -> List[Dict]:
        """Trouver les devices les plus suspects"""
        query = """
        MATCH (d:Device)<-[:USING]-(t:Transaction)<-[:MADE]-(a:Account)
        WITH d, 
             COUNT(DISTINCT a) as account_count,
             COUNT(t) as tx_count,
             SUM(CASE WHEN t.is_fraud = 1 THEN 1 ELSE 0 END) as fraud_count
        WHERE account_count >= $min_accounts
        RETURN d.id as device_id,
               account_count,
               tx_count,
               fraud_count,
               ROUND(100.0 * fraud_count / tx_count, 2) as fraud_rate
        ORDER BY account_count DESC, fraud_count DESC
        LIMIT 20
        """
        
        with self.driver.session() as session:
            result = session.run(query, min_accounts=min_accounts)
            return [dict(record) for record in result]


def main():
    """Démonstration du système"""
    
    # Initialiser GraphRAG
    graphrag = FraudGraphRAG()
    
    try:
        print("\n" + "="*70)
        print("🧪 DÉMONSTRATION DU SYSTÈME GRAPHRAG")
        print("="*70)
        
        # Test 1: Trouver des devices suspects
        print("\n📱 RECHERCHE DE DEVICES SUSPECTS...")
        suspicious_devices = graphrag.find_suspicious_devices(min_accounts=3)
        
        if suspicious_devices:
            print(f"\n✅ {len(suspicious_devices)} devices suspects trouvés:")
            for i, device in enumerate(suspicious_devices[:5], 1):
                print(f"\n{i}. Device: {device['device_id']}")
                print(f"   - Comptes: {device['account_count']}")
                print(f"   - Transactions: {device['tx_count']}")
                print(f"   - Fraudes: {device['fraud_count']} ({device['fraud_rate']:.1f}%)")
            
            # Test 2: Investiguer le premier device
            if suspicious_devices:
                device_id = suspicious_devices[0]['device_id']
                
                print("\n" + "="*70)
                print(f"🕵️ INVESTIGATION DU DEVICE: {device_id}")
                print("="*70)
                
                report = graphrag.investigate_fraud_network(device_id)
                print("\n" + report)
            
            # Test 3: Expliquer une transaction frauduleuse
            print("\n" + "="*70)
            print("🔍 EXPLICATION D'UNE TRANSACTION FRAUDULEUSE")
            print("="*70)
            
            # Trouver une transaction frauduleuse
            query = """
            MATCH (t:Transaction {is_fraud: 1})
            RETURN t.id as tx_id
            LIMIT 1
            """
            with graphrag.driver.session() as session:
                result = session.run(query)
                fraud_tx = [dict(record) for record in result]
            
            if fraud_tx:
                tx_id = fraud_tx[0]['tx_id']
                explanation = graphrag.explain_transaction(tx_id)
                print("\n" + explanation)
        
        else:
            print("❌ Aucun device suspect trouvé")
        
        print("\n" + "="*70)
        print("✅ DÉMONSTRATION TERMINÉE")
        print("="*70)
    
    finally:
        graphrag.close()


if __name__ == "__main__":
    main()