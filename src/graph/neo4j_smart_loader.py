"""
Smart Neo4j Loader - Charge par étapes avec vérifications
"""

from neo4j import GraphDatabase
from pathlib import Path
import pandas as pd
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"


class SmartNeo4jLoader:
    
    def __init__(self, uri: str, username: str, password: str):
        logger.info("🔌 Connexion à Neo4j...")
        self.driver = GraphDatabase.driver(
            uri, 
            auth=(username, password),
            max_connection_lifetime=7200,
            connection_timeout=120,
            max_transaction_retry_time=120
        )
        
        # Test de connexion
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1 AS test")
                result.single()
            logger.info("✅ Connexion réussie!")
        except Exception as e:
            logger.error(f"❌ Erreur de connexion: {e}")
            logger.info("\n💡 SOLUTION:")
            logger.info("   1. Vérifiez que Neo4j est démarré: docker ps")
            logger.info("   2. Attendez 30s après le démarrage")
            logger.info("   3. Testez la connexion: http://localhost:7474")
            raise
    
    def close(self):
        if self.driver:
            self.driver.close()
    
    def clear_all(self):
        """Supprime TOUT (nœuds + relations)"""
        logger.info("⚠️  Suppression de toutes les données...")
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("✅ Base vidée")
    
    def create_constraints(self):
        """Créer les contraintes"""
        logger.info("\n📋 Création des contraintes...")
        
        constraints = [
            "CREATE CONSTRAINT customer_id IF NOT EXISTS FOR (c:Customer) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT account_id IF NOT EXISTS FOR (a:Account) REQUIRE a.id IS UNIQUE",
            "CREATE CONSTRAINT transaction_id IF NOT EXISTS FOR (t:Transaction) REQUIRE t.id IS UNIQUE",
            "CREATE CONSTRAINT merchant_name IF NOT EXISTS FOR (m:Merchant) REQUIRE m.name IS UNIQUE",
            "CREATE CONSTRAINT device_id IF NOT EXISTS FOR (d:Device) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT location_id IF NOT EXISTS FOR (l:Location) REQUIRE l.id IS UNIQUE",
        ]
        
        with self.driver.session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                    logger.info(f"  ✅ Contrainte créée")
                except:
                    logger.info(f"  ⚠️  Contrainte existe déjà")
        
        logger.info("✅ Contraintes OK")
    
    def load_simple_nodes(self):
        """Charge les nœuds simples (pas les transactions)"""
        logger.info("\n📦 Chargement des nœuds simples...")
        
        # Customers
        logger.info("  👥 Customers...")
        customers_df = pd.read_csv(PROCESSED_DATA_DIR / "neo4j_customers.csv").where(pd.notna, None)
        self._load_in_batches(
            data=customers_df.to_dict('records'),
            query="""
            UNWIND $batch AS c
            CREATE (n:Customer {
                id: c.customer_id,
                first_name: c.first,
                last_name: c.last,
                gender: c.gender,
                age: c.customer_age
            })
            """,
            batch_size=500,
            name="Customers"
        )
        
        # Accounts
        logger.info("  💳 Accounts...")
        accounts_df = pd.read_csv(PROCESSED_DATA_DIR / "neo4j_accounts.csv").where(pd.notna, None)
        self._load_in_batches(
            data=accounts_df.to_dict('records'),
            query="""
            UNWIND $batch AS a
            CREATE (n:Account {
                id: a.account_id,
                cc_number: a.cc_num,
                customer_id: a.customer_id
            })
            """,
            batch_size=500,
            name="Accounts"
        )
        
        # Merchants
        logger.info("  🏪 Merchants...")
        merchants_df = pd.read_csv(PROCESSED_DATA_DIR / "neo4j_merchants.csv").where(pd.notna, None)
        self._load_in_batches(
            data=merchants_df.to_dict('records'),
            query="""
            UNWIND $batch AS m
            CREATE (n:Merchant {
                name: m.merchant,
                category: m.category
            })
            """,
            batch_size=500,
            name="Merchants"
        )
        
        # Devices
        logger.info("  📱 Devices...")
        devices_df = pd.read_csv(PROCESSED_DATA_DIR / "neo4j_devices.csv").where(pd.notna, None)
        self._load_in_batches(
            data=devices_df.to_dict('records'),
            query="""
            UNWIND $batch AS d
            CREATE (n:Device {id: d.device_id})
            """,
            batch_size=1000,
            name="Devices"
        )
        
        # Locations
        logger.info("  📍 Locations...")
        locations_df = pd.read_csv(PROCESSED_DATA_DIR / "neo4j_locations.csv").where(pd.notna, None)
        self._load_in_batches(
            data=locations_df.to_dict('records'),
            query="""
            UNWIND $batch AS l
            CREATE (n:Location {
                id: l.location_id,
                city: l.city,
                state: l.state
            })
            """,
            batch_size=500,
            name="Locations"
        )
        
        logger.info("✅ Nœuds simples chargés!")
    
    def load_transactions_with_relationships(self):
        """Charge les transactions ET crée les relations en même temps"""
        logger.info("\n💰 Chargement des transactions avec relations...")
        
        transactions_df = pd.read_csv(PROCESSED_DATA_DIR / "neo4j_transactions.csv")
        transactions_df = transactions_df.where(pd.notna(transactions_df), None)
        transactions_df['trans_date_trans_time'] = pd.to_datetime(
            transactions_df['trans_date_trans_time']
        ).dt.strftime('%Y-%m-%dT%H:%M:%S')
        
        data = transactions_df.to_dict('records')
        
        # Query qui crée la transaction ET les relations
        query = """
        UNWIND $batch AS tx
        
        // Créer la transaction
        CREATE (t:Transaction {
            id: tx.transaction_id,
            trans_num: tx.trans_num,
            timestamp: datetime(tx.trans_date_trans_time),
            amount: tx.amt,
            is_fraud: tx.is_fraud,
            category: tx.category
        })
        
        // Créer les relations
        WITH t, tx
        MATCH (a:Account {id: tx.account_id})
        MATCH (m:Merchant {name: tx.merchant})
        MATCH (d:Device {id: tx.device_id})
        MATCH (l:Location {id: tx.location_id})
        
        CREATE (a)-[:MADE]->(t)
        CREATE (t)-[:AT]->(m)
        CREATE (t)-[:USING]->(d)
        CREATE (t)-[:IN]->(l)
        """
        
        self._load_in_batches(
            data=data,
            query=query,
            batch_size=200,  # Plus petit car beaucoup de MATCH
            name="Transactions + Relations"
        )
        
        logger.info("✅ Transactions et relations chargées!")
    
    def create_customer_relationships(self):
        """Créer la relation Customer → Account"""
        logger.info("\n🔗 Création: Customer → Account...")
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (a:Account)
                MATCH (c:Customer {id: a.customer_id})
                MERGE (c)-[:OWNS]->(a)
            """)
            summary = result.consume()
            count = summary.counters.relationships_created
            logger.info(f"✅ {count} relations OWNS créées")
    
    def _load_in_batches(self, data: list, query: str, batch_size: int, name: str):
        """Helper pour charger par batches"""
        total = len(data)
        start_time = time.time()
        
        with self.driver.session() as session:
            for i in range(0, total, batch_size):
                batch = data[i:i+batch_size]
                session.run(query, batch=batch)
                
                processed = min(i+batch_size, total)
                if processed % 10000 == 0 or processed == total:
                    elapsed = time.time() - start_time
                    rate = processed / elapsed if elapsed > 0 else 0
                    remaining = (total - processed) / rate if rate > 0 else 0
                    logger.info(f"    📊 {processed:,}/{total:,} ({rate:.0f}/s, ~{remaining/60:.1f} min restantes)")
        
        elapsed = time.time() - start_time
        logger.info(f"  ✅ {name}: {total:,} en {elapsed:.1f}s")
    
    def verify(self):
        """Vérification rapide"""
        logger.info("\n🔍 Vérification...")
        
        with self.driver.session() as session:
            # Compter tout
            stats = {}
            for label in ['Customer', 'Account', 'Transaction', 'Merchant', 'Device', 'Location']:
                result = session.run(f"MATCH (n:{label}) RETURN count(n) as count")
                stats[label] = result.single()['count']
            
            for label, count in stats.items():
                logger.info(f"  • {label}: {count:,}")
            
            # Relations
            result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            rel_count = result.single()['count']
            logger.info(f"  • Relations: {rel_count:,}")
        
        logger.info("✅ Vérification terminée!")


def main():
    print("="*70)
    print("🚀 SMART NEO4J LOADER")
    print("="*70)
    
    loader = SmartNeo4jLoader(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="fraudpassword"
    )
    
    try:
        start = time.time()
        
        # Pipeline
        loader.clear_all()
        loader.create_constraints()
        loader.load_simple_nodes()
        loader.load_transactions_with_relationships()
        loader.create_customer_relationships()
        loader.verify()
        
        elapsed = time.time() - start
        print(f"\n✅ TERMINÉ EN {elapsed/60:.1f} MINUTES!")
        
    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        raise
    finally:
        loader.close()


if __name__ == "__main__":
    main()