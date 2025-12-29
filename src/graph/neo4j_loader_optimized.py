"""
Neo4j Loader OPTIMISÉ - Version Rapide
======================================
Cette version crée les relations lors du chargement des transactions
pour éviter les timeouts et produits cartésiens.
"""

from neo4j import GraphDatabase
from pathlib import Path
import pandas as pd
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"


class Neo4jFraudGraphLoaderOptimized:
    """Loader optimisé avec création de relations en batch"""
    
    def __init__(self, uri: str, username: str, password: str):
        logger.info("🔌 Connexion à Neo4j...")
        
        # Configuration avec timeout augmenté
        self.driver = GraphDatabase.driver(
            uri, 
            auth=(username, password),
            max_connection_lifetime=3600,  # 1 heure
            connection_timeout=60,          # 60 secondes
            max_transaction_retry_time=60   # 60 secondes
        )
        
        # Test de connexion
        with self.driver.session() as session:
            result = session.run("RETURN 1 AS test")
            result.single()
        
        logger.info("✅ Connexion réussie à Neo4j!")
    
    def close(self):
        if self.driver:
            self.driver.close()
            logger.info("🔌 Connexion fermée")
    
    def clear_relationships_only(self):
        """
        Supprime UNIQUEMENT les relations, pas les nœuds
        Utile si vous avez déjà chargé les nœuds
        """
        logger.info("⚠️  Suppression des relations existantes...")
        with self.driver.session() as session:
            session.run("MATCH ()-[r]->() DELETE r")
        logger.info("✅ Relations supprimées")
    
    def create_relationships_optimized(self):
        """
        Créer les relations de manière OPTIMISÉE
        
        STRATÉGIE:
        - Utiliser des batch plus petits
        - Créer les relations en utilisant les IDs directement
        - Éviter les produits cartésiens
        """
        logger.info("\n🔗 Création des relations (version optimisée)...")
        
        # Charger le fichier des transactions
        transactions_df = pd.read_csv(PROCESSED_DATA_DIR / "neo4j_transactions.csv")
        transactions_df = transactions_df.where(pd.notna(transactions_df), None)
        
        total = len(transactions_df)
        batch_size = 1000
        
        logger.info(f"📊 {total} relations à créer...")
        
        # Query optimisée: utilise les IDs directement
        query = """
        UNWIND $batch AS row
        
        // MADE: Account → Transaction
        MATCH (a:Account {id: row.account_id})
        MATCH (t:Transaction {id: row.transaction_id})
        MERGE (a)-[:MADE]->(t)
        
        // AT: Transaction → Merchant
        WITH row, t
        MATCH (m:Merchant {name: row.merchant})
        MERGE (t)-[:AT]->(m)
        
        // USING: Transaction → Device
        WITH row, t
        MATCH (d:Device {id: row.device_id})
        MERGE (t)-[:USING]->(d)
        
        // IN: Transaction → Location
        WITH row, t
        MATCH (l:Location {id: row.location_id})
        MERGE (t)-[:IN]->(l)
        """
        
        start_time = time.time()
        
        with self.driver.session() as session:
            for i in range(0, total, batch_size):
                # Préparer le batch
                batch_df = transactions_df.iloc[i:i+batch_size]
                batch = []
                
                for _, row in batch_df.iterrows():
                    batch.append({
                        'transaction_id': f"TX_{row['trans_num']}",
                        'account_id': row['account_id'],
                        'merchant': row['merchant'],
                        'device_id': row['device_id'],
                        'location_id': row['location_id']
                    })
                
                # Exécuter
                session.run(query, batch=batch)
                
                # Progress
                processed = min(i+batch_size, total)
                if processed % 10000 == 0 or processed == total:
                    elapsed = time.time() - start_time
                    rate = processed / elapsed if elapsed > 0 else 0
                    remaining = (total - processed) / rate if rate > 0 else 0
                    
                    logger.info(
                        f"  📊 {processed:,}/{total:,} "
                        f"({rate:.0f} tx/s, "
                        f"~{remaining/60:.1f} min restantes)"
                    )
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Toutes les relations créées en {elapsed/60:.1f} minutes!")
    
    def create_customer_account_relationships(self):
        """
        Créer uniquement la relation OWNS (Customer → Account)
        C'est rapide car il y a peu de comptes
        """
        logger.info("\n🔗 Création: OWNS (Customer → Account)...")
        
        query = """
        MATCH (a:Account)
        MATCH (c:Customer {id: a.customer_id})
        MERGE (c)-[:OWNS]->(a)
        """
        
        with self.driver.session() as session:
            result = session.run(query)
            summary = result.consume()
            count = summary.counters.relationships_created
            logger.info(f"✅ {count} relations OWNS créées")
    
    def verify_graph(self):
        """Vérifier l'intégrité du graphe"""
        logger.info("\n🔍 Vérification du graphe...")
        
        with self.driver.session() as session:
            # Compter les nœuds
            logger.info("\n📊 Nœuds:")
            for label in ['Customer', 'Account', 'Transaction', 'Merchant', 'Device', 'Location']:
                result = session.run(f"MATCH (n:{label}) RETURN count(n) as count")
                count = result.single()['count']
                logger.info(f"  • {label}: {count:,}")
            
            # Compter les relations
            logger.info("\n🔗 Relations:")
            for rel_type in ['OWNS', 'MADE', 'AT', 'USING', 'IN']:
                result = session.run(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) as count")
                count = result.single()['count']
                logger.info(f"  • {rel_type}: {count:,}")
            
            # Statistiques de fraude
            logger.info("\n📈 Statistiques:")
            
            # Devices suspects
            result = session.run("""
                MATCH (d:Device)<-[:USING]-(t:Transaction)<-[:MADE]-(a:Account)
                WITH d, COUNT(DISTINCT a) as account_count
                WHERE account_count > 1
                RETURN COUNT(d) as suspicious_devices, 
                       MAX(account_count) as max_accounts
            """)
            record = result.single()
            logger.info(f"  🚨 Devices suspects: {record['suspicious_devices']:,} "
                       f"(max {record['max_accounts']} comptes sur 1 device)")
            
            # Transactions frauduleuses
            result = session.run("""
                MATCH (t:Transaction)
                WITH COUNT(t) as total,
                     SUM(CASE WHEN t.is_fraud = 1 THEN 1 ELSE 0 END) as fraud
                RETURN total, fraud, 
                       ROUND(100.0 * fraud / total, 2) as fraud_rate
            """)
            record = result.single()
            logger.info(f"  💰 Transactions: {record['total']:,} total, "
                       f"{record['fraud']:,} frauduleuses ({record['fraud_rate']}%)")
        
        logger.info("\n✅ Vérification terminée!")
    
    def run_sample_fraud_queries(self):
        """Requêtes de détection de fraude"""
        logger.info("\n🎯 Requêtes de détection de fraude...")
        
        with self.driver.session() as session:
            # Query 1: Top devices suspects
            logger.info("\n📱 Top 5 devices les plus suspects:")
            result = session.run("""
                MATCH (d:Device)<-[:USING]-(t:Transaction)<-[:MADE]-(a:Account)
                WITH d, 
                     COUNT(DISTINCT a) as account_count,
                     COUNT(t) as tx_count,
                     SUM(CASE WHEN t.is_fraud = 1 THEN 1 ELSE 0 END) as fraud_count
                WHERE account_count > 1
                RETURN d.id as device_id,
                       account_count,
                       tx_count,
                       fraud_count,
                       ROUND(100.0 * fraud_count / tx_count, 1) as fraud_rate
                ORDER BY account_count DESC, fraud_count DESC
                LIMIT 5
            """)
            
            for i, record in enumerate(result, 1):
                logger.info(f"  {i}. Device: {record['device_id']}")
                logger.info(f"     └─ {record['account_count']} comptes | "
                          f"{record['tx_count']} tx | "
                          f"{record['fraud_count']} frauds ({record['fraud_rate']}%)")
            
            # Query 2: Réseau d'un device suspect
            logger.info("\n🕸️  Réseau autour d'un device suspect:")
            result = session.run("""
                MATCH (d:Device)<-[:USING]-(t:Transaction)<-[:MADE]-(a:Account)
                WITH d, COUNT(DISTINCT a) as account_count
                WHERE account_count > 3
                WITH d LIMIT 1
                
                MATCH (d)<-[:USING]-(t:Transaction)<-[:MADE]-(a:Account)
                      <-[:OWNS]-(c:Customer)
                RETURN d.id as device_id,
                       COLLECT(DISTINCT {
                           customer: c.first_name + ' ' + c.last_name,
                           account: a.id,
                           tx_count: COUNT(t),
                           fraud_count: SUM(CASE WHEN t.is_fraud = 1 THEN 1 ELSE 0 END)
                       }) as connected_accounts
                LIMIT 1
            """)
            
            record = result.single()
            if record:
                logger.info(f"  Device: {record['device_id']}")
                logger.info(f"  Comptes connectés:")
                for acc in record['connected_accounts'][:5]:  # Limiter à 5
                    logger.info(f"    • {acc['customer']} ({acc['account']})")
                    logger.info(f"      └─ {acc['tx_count']} tx, {acc['fraud_count']} frauds")
            
            # Query 3: Merchants à risque
            logger.info("\n🏪 Top 5 merchants avec le plus de fraudes:")
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
            
            for i, record in enumerate(result, 1):
                logger.info(f"  {i}. {record['merchant']} ({record['category']})")
                logger.info(f"     └─ {record['fraud_tx']}/{record['total_tx']} tx frauduleuses "
                          f"({record['fraud_rate']}%)")
        
        logger.info("\n✅ Requêtes terminées!")


def main():
    print("="*70)
    print("🚀 NEO4J FRAUD GRAPH - VERSION OPTIMISÉE")
    print("="*70)
    
    # Configuration
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USERNAME = "neo4j"
    NEO4J_PASSWORD = "fraudpassword"
    
    loader = Neo4jFraudGraphLoaderOptimized(
        uri=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD
    )
    
    try:
        start_time = time.time()
        
        # IMPORTANT: Comme vous avez déjà les nœuds, on supprime juste les relations
        loader.clear_relationships_only()
        
        # Créer les relations de manière optimisée
        loader.create_customer_account_relationships()  # Rapide
        loader.create_relationships_optimized()         # Plus long mais optimisé
        
        # Vérifier
        loader.verify_graph()
        
        # Requêtes de test
        loader.run_sample_fraud_queries()
        
        elapsed = time.time() - start_time
        
        print("\n" + "="*70)
        print(f"✅ TERMINÉ EN {elapsed/60:.1f} MINUTES!")
        print("="*70)
        print("\n🎉 Votre Knowledge Graph est maintenant complet!")
        print(f"\n📊 Neo4j Browser: http://localhost:7474")
        
    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        raise
    finally:
        loader.close()


if __name__ == "__main__":
    main()