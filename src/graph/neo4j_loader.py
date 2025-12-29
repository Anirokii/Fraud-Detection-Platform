"""
Neo4j Knowledge Graph Loader
=============================
Ce script charge les données enrichies dans Neo4j pour créer
un Knowledge Graph permettant la détection de réseaux de fraude.

Architecture du Graphe:
- Nodes: Customer, Account, Transaction, Merchant, Device, Location
- Relationships: OWNS, MADE, AT, USING, IN
"""

from neo4j import GraphDatabase
from pathlib import Path
import pandas as pd
import time
from typing import Dict, List
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Chemins
BASE_DIR = Path(__file__).parent.parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"


class Neo4jFraudGraphLoader:
    """Classe pour charger les données de fraude dans Neo4j"""
    
    def __init__(self, uri: str, username: str, password: str):
        """
        Initialise la connexion Neo4j
        
        Args:
            uri: URI de connexion Neo4j (ex: bolt://localhost:7687)
            username: Nom d'utilisateur
            password: Mot de passe
        """
        logger.info("🔌 Connexion à Neo4j...")
        try:
            self.driver = GraphDatabase.driver(uri, auth=(username, password))
            # Test de connexion
            with self.driver.session() as session:
                result = session.run("RETURN 1 AS test")
                result.single()
            logger.info("✅ Connexion réussie à Neo4j!")
        except Exception as e:
            logger.error(f"❌ Erreur de connexion: {e}")
            raise
    
    def close(self):
        """Ferme la connexion"""
        if self.driver:
            self.driver.close()
            logger.info("🔌 Connexion fermée")
    
    def clear_database(self):
        """
        ATTENTION: Supprime TOUTES les données de la base!
        Utilisé pour repartir de zéro.
        """
        logger.warning("⚠️  Suppression de toutes les données...")
        with self.driver.session() as session:
            # Supprimer tous les nœuds et relations
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("✅ Base de données vidée")
    
    def create_constraints_and_indexes(self):
        """
        Étape 1: Créer les contraintes et index
        
        POURQUOI?
        - Contraintes: Garantissent l'unicité des IDs (pas de doublons)
        - Index: Accélèrent les recherches (performance)
        """
        logger.info("\n📋 Création des contraintes et index...")
        
        constraints = [
            # Contraintes d'unicité
            "CREATE CONSTRAINT customer_id IF NOT EXISTS FOR (c:Customer) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT account_id IF NOT EXISTS FOR (a:Account) REQUIRE a.id IS UNIQUE",
            "CREATE CONSTRAINT transaction_id IF NOT EXISTS FOR (t:Transaction) REQUIRE t.id IS UNIQUE",
            "CREATE CONSTRAINT merchant_id IF NOT EXISTS FOR (m:Merchant) REQUIRE m.name IS UNIQUE",
            "CREATE CONSTRAINT device_id IF NOT EXISTS FOR (d:Device) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT location_id IF NOT EXISTS FOR (l:Location) REQUIRE l.id IS UNIQUE",
        ]
        
        with self.driver.session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                    logger.info(f"✅ {constraint.split()[1]} créé")
                except Exception as e:
                    logger.warning(f"⚠️  {constraint.split()[1]} existe déjà")
        
        logger.info("✅ Contraintes et index créés!")
    
    def load_customers(self):
        """
        Étape 2: Charger les CUSTOMERS
        
        Structure:
        (:Customer {
            id: 'customer_id',
            first_name: 'John',
            last_name: 'Doe',
            gender: 'M',
            age: 35,
            ...
        })
        """
        logger.info("\n👥 Chargement des Customers...")
        
        # Lire le fichier CSV
        customers_df = pd.read_csv(PROCESSED_DATA_DIR / "neo4j_customers.csv")
        
        # Remplacer NaN par None pour Neo4j
        customers_df = customers_df.where(pd.notna(customers_df), None)
        
        # Convertir en liste de dictionnaires
        customers = customers_df.to_dict('records')
        
        # Query Cypher pour créer les nœuds
        query = """
        UNWIND $customers AS customer
        CREATE (c:Customer {
            id: customer.customer_id,
            first_name: customer.first,
            last_name: customer.last,
            gender: customer.gender,
            street: customer.street,
            city: customer.city,
            state: customer.state,
            zip: customer.zip,
            date_of_birth: customer.dob,
            age: customer.customer_age
        })
        """
        
        # Charger par batch de 1000 pour performance
        batch_size = 1000
        total = len(customers)
        
        with self.driver.session() as session:
            for i in range(0, total, batch_size):
                batch = customers[i:i+batch_size]
                session.run(query, customers=batch)
                logger.info(f"  📊 {min(i+batch_size, total)}/{total} customers chargés")
        
        logger.info(f"✅ {total} Customers chargés!")
    
    def load_accounts(self):
        """
        Étape 3: Charger les ACCOUNTS
        """
        logger.info("\n💳 Chargement des Accounts...")
        
        accounts_df = pd.read_csv(PROCESSED_DATA_DIR / "neo4j_accounts.csv")
        accounts_df = accounts_df.where(pd.notna(accounts_df), None)
        accounts = accounts_df.to_dict('records')
        
        query = """
        UNWIND $accounts AS account
        CREATE (a:Account {
            id: account.account_id,
            cc_number: account.cc_num,
            customer_id: account.customer_id
        })
        """
        
        batch_size = 1000
        total = len(accounts)
        
        with self.driver.session() as session:
            for i in range(0, total, batch_size):
                batch = accounts[i:i+batch_size]
                session.run(query, accounts=batch)
                logger.info(f"  📊 {min(i+batch_size, total)}/{total} accounts chargés")
        
        logger.info(f"✅ {total} Accounts chargés!")
    
    def load_merchants(self):
        """
        Étape 4: Charger les MERCHANTS
        """
        logger.info("\n🏪 Chargement des Merchants...")
        
        merchants_df = pd.read_csv(PROCESSED_DATA_DIR / "neo4j_merchants.csv")
        merchants_df = merchants_df.where(pd.notna(merchants_df), None)
        merchants = merchants_df.to_dict('records')
        
        query = """
        UNWIND $merchants AS merchant
        CREATE (m:Merchant {
            name: merchant.merchant,
            category: merchant.category,
            merchant_id: merchant.merchant_id
        })
        """
        
        batch_size = 1000
        total = len(merchants)
        
        with self.driver.session() as session:
            for i in range(0, total, batch_size):
                batch = merchants[i:i+batch_size]
                session.run(query, merchants=batch)
                logger.info(f"  📊 {min(i+batch_size, total)}/{total} merchants chargés")
        
        logger.info(f"✅ {total} Merchants chargés!")
    
    def load_devices(self):
        """
        Étape 5: Charger les DEVICES
        """
        logger.info("\n📱 Chargement des Devices...")
        
        devices_df = pd.read_csv(PROCESSED_DATA_DIR / "neo4j_devices.csv")
        devices_df = devices_df.where(pd.notna(devices_df), None)
        devices = devices_df.to_dict('records')
        
        query = """
        UNWIND $devices AS device
        CREATE (d:Device {
            id: device.device_id
        })
        """
        
        batch_size = 10000
        total = len(devices)
        
        with self.driver.session() as session:
            for i in range(0, total, batch_size):
                batch = devices[i:i+batch_size]
                session.run(query, devices=batch)
                logger.info(f"  📊 {min(i+batch_size, total)}/{total} devices chargés")
        
        logger.info(f"✅ {total} Devices chargés!")
    
    def load_locations(self):
        """
        Étape 6: Charger les LOCATIONS
        """
        logger.info("\n📍 Chargement des Locations...")
        
        locations_df = pd.read_csv(PROCESSED_DATA_DIR / "neo4j_locations.csv")
        locations_df = locations_df.where(pd.notna(locations_df), None)
        locations = locations_df.to_dict('records')
        
        query = """
        UNWIND $locations AS location
        CREATE (l:Location {
            id: location.location_id,
            city: location.city,
            state: location.state,
            latitude: location.lat,
            longitude: location.long
        })
        """
        
        batch_size = 1000
        total = len(locations)
        
        with self.driver.session() as session:
            for i in range(0, total, batch_size):
                batch = locations[i:i+batch_size]
                session.run(query, locations=batch)
                logger.info(f"  📊 {min(i+batch_size, total)}/{total} locations chargées")
        
        logger.info(f"✅ {total} Locations chargées!")
    
    def load_transactions(self):
        """
        Étape 7: Charger les TRANSACTIONS
        
        IMPORTANT: Les transactions sont le cœur du graphe!
        Elles contiennent toutes les features de fraude.
        """
        logger.info("\n💰 Chargement des Transactions...")
        
        transactions_df = pd.read_csv(PROCESSED_DATA_DIR / "neo4j_transactions.csv")
        
        # Remplacer NaN par None
        transactions_df = transactions_df.where(pd.notna(transactions_df), None)
        
        # Convertir les dates en string ISO
        transactions_df['trans_date_trans_time'] = pd.to_datetime(
            transactions_df['trans_date_trans_time']
        ).dt.strftime('%Y-%m-%dT%H:%M:%S')
        
        transactions = transactions_df.to_dict('records')
        
        query = """
        UNWIND $transactions AS tx
        CREATE (t:Transaction {
            id: tx.transaction_id,
            trans_num: tx.trans_num,
            timestamp: datetime(tx.trans_date_trans_time),
            amount: tx.amt,
            is_fraud: tx.is_fraud,
            category: tx.category,
            hour: tx.hour,
            day_of_week: tx.day_of_week,
            is_weekend: tx.is_weekend,
            is_night: tx.is_night,
            tx_velocity: tx.tx_velocity,
            amt_deviation: tx.amt_deviation,
            is_high_risk_category: tx.is_high_risk_category,
            customer_tx_count: tx.customer_tx_count,
            days_since_first_tx: tx.days_since_first_tx,
            account_id: tx.account_id,
            merchant_name: tx.merchant,
            device_id: tx.device_id,
            location_id: tx.location_id
        })
        """
        
        batch_size = 2000 # Plus petit car transactions sont volumineuses
        total = len(transactions)
        
        logger.info(f"  ⏳ Chargement de {total} transactions (cela peut prendre quelques minutes)...")
        
        with self.driver.session() as session:
            start_time = time.time()
            for i in range(0, total, batch_size):
                batch = transactions[i:i+batch_size]
                session.run(query, transactions=batch)
                
                # Afficher progression toutes les 10000 transactions
                if (i+batch_size) % 10000 == 0 or i+batch_size >= total:
                    elapsed = time.time() - start_time
                    rate = (i+batch_size) / elapsed
                    logger.info(f"  📊 {min(i+batch_size, total)}/{total} transactions ({rate:.0f} tx/s)")
        
        logger.info(f"✅ {total} Transactions chargées!")
    
    def create_relationships(self):
        """
        Étape 8: Créer les RELATIONS entre les nœuds
        
        C'EST ICI QUE LE GRAPHE PREND VIE!
        Les relations permettent de traverser le réseau.
        """
        logger.info("\n🔗 Création des relations...")
        
        relationships = [
            {
                'name': 'OWNS (Customer → Account)',
                'query': """
                MATCH (c:Customer), (a:Account)
                WHERE c.id = a.customer_id
                CREATE (c)-[:OWNS]->(a)
                """
            },
            {
                'name': 'MADE (Account → Transaction)',
                'query': """
                MATCH (a:Account), (t:Transaction)
                WHERE a.id = t.account_id
                CREATE (a)-[:MADE]->(t)
                """
            },
            {
                'name': 'AT (Transaction → Merchant)',
                'query': """
                MATCH (t:Transaction), (m:Merchant)
                WHERE t.merchant_name = m.name
                CREATE (t)-[:AT]->(m)
                """
            },
            {
                'name': 'USING (Transaction → Device)',
                'query': """
                MATCH (t:Transaction), (d:Device)
                WHERE t.device_id = d.id
                CREATE (t)-[:USING]->(d)
                """
            },
            {
                'name': 'IN (Transaction → Location)',
                'query': """
                MATCH (t:Transaction), (l:Location)
                WHERE t.location_id = l.id
                CREATE (t)-[:IN]->(l)
                """
            }
        ]
        
        with self.driver.session() as session:
            for rel in relationships:
                logger.info(f"  🔗 Création: {rel['name']}...")
                start_time = time.time()
                result = session.run(rel['query'])
                elapsed = time.time() - start_time
                
                # Compter les relations créées
                summary = result.consume()
                count = summary.counters.relationships_created
                
                logger.info(f"  ✅ {count} relations créées en {elapsed:.1f}s")
        
        logger.info("✅ Toutes les relations créées!")
    
    def verify_graph(self) -> Dict:
        """
        Étape 9: Vérifier l'intégrité du graphe
        
        Compte les nœuds et relations pour s'assurer que tout est chargé.
        """
        logger.info("\n🔍 Vérification du graphe...")
        
        with self.driver.session() as session:
            # Compter les nœuds
            node_counts = {}
            for label in ['Customer', 'Account', 'Transaction', 'Merchant', 'Device', 'Location']:
                result = session.run(f"MATCH (n:{label}) RETURN count(n) as count")
                count = result.single()['count']
                node_counts[label] = count
                logger.info(f"  📊 {label}: {count:,} nœuds")
            
            # Compter les relations
            rel_counts = {}
            for rel_type in ['OWNS', 'MADE', 'AT', 'USING', 'IN']:
                result = session.run(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) as count")
                count = result.single()['count']
                rel_counts[rel_type] = count
                logger.info(f"  🔗 {rel_type}: {count:,} relations")
            
            # Statistiques supplémentaires
            logger.info("\n📈 Statistiques du graphe:")
            
            # Transactions frauduleuses
            result = session.run("MATCH (t:Transaction {is_fraud: 1}) RETURN count(t) as count")
            fraud_count = result.single()['count']
            fraud_rate = (fraud_count / node_counts['Transaction']) * 100
            logger.info(f"  🚨 Transactions frauduleuses: {fraud_count:,} ({fraud_rate:.2f}%)")
            
            # Devices suspects (partagés)
            result = session.run("""
                MATCH (d:Device)<-[:USING]-(t:Transaction)<-[:MADE]-(a:Account)
                WITH d, COUNT(DISTINCT a) as account_count
                WHERE account_count > 1
                RETURN COUNT(d) as suspicious_devices
            """)
            suspicious_devices = result.single()['suspicious_devices']
            logger.info(f"  ⚠️  Devices suspects (>1 account): {suspicious_devices:,}")
            
            # Merchants à risque
            result = session.run("""
                MATCH (m:Merchant)<-[:AT]-(t:Transaction {is_fraud: 1})
                WITH m, COUNT(t) as fraud_count
                WHERE fraud_count > 5
                RETURN COUNT(m) as risky_merchants
            """)
            risky_merchants = result.single()['risky_merchants']
            logger.info(f"  🏪 Merchants à risque (>5 frauds): {risky_merchants:,}")
        
        logger.info("\n✅ Vérification terminée!")
        
        return {
            'nodes': node_counts,
            'relationships': rel_counts,
            'fraud_count': fraud_count,
            'fraud_rate': fraud_rate,
            'suspicious_devices': suspicious_devices,
            'risky_merchants': risky_merchants
        }
    
    def run_sample_queries(self):
        """
        Étape 10: Exécuter des requêtes de test
        
        Démonstration de la puissance du graphe!
        """
        logger.info("\n🎯 Exécution de requêtes de test...")
        
        with self.driver.session() as session:
            # Query 1: Trouver un device suspect
            logger.info("\n📱 Query 1: Top 5 devices les plus suspects")
            result = session.run("""
                MATCH (d:Device)<-[:USING]-(t:Transaction)<-[:MADE]-(a:Account)
                WITH d, COUNT(DISTINCT a) as account_count, COUNT(t) as tx_count
                WHERE account_count > 1
                RETURN d.id as device_id, 
                       account_count, 
                       tx_count
                ORDER BY account_count DESC
                LIMIT 5
            """)
            
            for record in result:
                logger.info(f"  🚨 Device: {record['device_id']}")
                logger.info(f"     └─ {record['account_count']} comptes, {record['tx_count']} transactions")
            
            # Query 2: Réseau autour d'un device suspect
            logger.info("\n🕸️  Query 2: Réseau d'un device suspect")
            result = session.run("""
                MATCH (d:Device)<-[:USING]-(t:Transaction)<-[:MADE]-(a:Account)
                WITH d, COUNT(DISTINCT a) as account_count
                WHERE account_count > 2
                WITH d
                LIMIT 1
                MATCH (d)<-[:USING]-(t:Transaction)<-[:MADE]-(a:Account)<-[:OWNS]-(c:Customer)
                RETURN d.id as device_id,
                       c.first_name + ' ' + c.last_name as customer_name,
                       a.id as account_id,
                       COUNT(t) as tx_count,
                       SUM(CASE WHEN t.is_fraud = 1 THEN 1 ELSE 0 END) as fraud_count
            """)
            
            logger.info(f"  Comptes connectés:")
            for record in result:
                logger.info(f"    • {record['customer_name']} ({record['account_id']})")
                logger.info(f"      └─ {record['tx_count']} tx, {record['fraud_count']} frauds")
            
            # Query 3: Merchants les plus risqués
            logger.info("\n🏪 Query 3: Top 5 merchants avec le plus de fraudes")
            result = session.run("""
                MATCH (m:Merchant)<-[:AT]-(t:Transaction)
                WITH m, 
                     COUNT(t) as total_tx,
                     SUM(CASE WHEN t.is_fraud = 1 THEN 1 ELSE 0 END) as fraud_tx
                WHERE fraud_tx > 0
                RETURN m.name as merchant,
                       m.category as category,
                       total_tx,
                       fraud_tx,
                       ROUND(100.0 * fraud_tx / total_tx, 2) as fraud_rate
                ORDER BY fraud_tx DESC
                LIMIT 5
            """)
            
            for record in result:
                logger.info(f"  🏪 {record['merchant']} ({record['category']})")
                logger.info(f"     └─ {record['fraud_tx']}/{record['total_tx']} frauds ({record['fraud_rate']}%)")
        
        logger.info("\n✅ Requêtes de test terminées!")


def main():
    """
    Point d'entrée principal
    """
    print("="*70)
    print("🚀 NEO4J FRAUD KNOWLEDGE GRAPH LOADER")
    print("="*70)
    
    # Configuration Neo4j
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USERNAME = "neo4j"
    NEO4J_PASSWORD = "fraudpassword"
    
    # Créer le loader
    loader = Neo4jFraudGraphLoader(
        uri=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD
    )
    
    try:
        # Pipeline complet
        start_time = time.time()
        
        # Optionnel: Vider la base (commentez si vous voulez garder les données)
        loader.clear_database()
        
        # Étape 1: Contraintes et index
        loader.create_constraints_and_indexes()
        
        # Étape 2-6: Charger les nœuds
        loader.load_customers()
        loader.load_accounts()
        loader.load_merchants()
        loader.load_devices()
        loader.load_locations()
        loader.load_transactions()
        
        # Étape 7: Créer les relations
        loader.create_relationships()
        
        # Étape 8: Vérifier
        stats = loader.verify_graph()
        
        # Étape 9: Requêtes de test
        loader.run_sample_queries()
        
        elapsed = time.time() - start_time
        
        print("\n" + "="*70)
        print(f"✅ CHARGEMENT TERMINÉ EN {elapsed:.1f}s!")
        print("="*70)
        print("\n🎉 Votre Knowledge Graph est prêt!")
        print(f"\n📊 Accédez à Neo4j Browser: http://localhost:7474")
        print(f"   Username: {NEO4J_USERNAME}")
        print(f"   Password: {NEO4J_PASSWORD}")
        
    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        raise
    finally:
        loader.close()


if __name__ == "__main__":
    main()