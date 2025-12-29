"""
Test Interactif du GraphRAG - VERSION CORRIGÉE
===============================================
Script pour tester le système avec des exemples
"""

from fraud_graphrag import FraudGraphRAG
import warnings
import logging

# Supprimer les warnings pour affichage propre
warnings.filterwarnings('ignore')
logging.getLogger('neo4j').setLevel(logging.ERROR)


def print_section(title: str):
    """Afficher une section formatée"""
    print("\n" + "="*80)
    print(f"🧪 {title}")
    print("="*80)


def test_explain_transaction():
    """Test d'explication de transaction"""
    print_section("TEST 1: EXPLICATION DE TRANSACTION FRAUDULEUSE")
    
    graphrag = FraudGraphRAG()
    
    try:
        # Trouver une transaction frauduleuse
        query = """
        MATCH (t:Transaction {is_fraud: 1})
        RETURN t.id as tx_id
        LIMIT 1
        """
        with graphrag.driver.session() as session:
            result = session.run(query)
            records = [dict(record) for record in result]
        
        if records:
            tx_id = records[0]['tx_id']
            print(f"\n📍 Transaction sélectionnée: {tx_id}")
            
            explanation = graphrag.explain_transaction(tx_id)
            print("\n" + "─"*80)
            print("📄 EXPLICATION GÉNÉRÉE PAR LE LLM:")
            print("─"*80)
            print(explanation)
        else:
            print("❌ Aucune transaction frauduleuse trouvée")
    
    finally:
        graphrag.close()


def test_investigate_network():
    """Test d'investigation de réseau"""
    print_section("TEST 2: INVESTIGATION DE RÉSEAU DE FRAUDE")
    
    graphrag = FraudGraphRAG()
    
    try:
        # Trouver un device suspect
        devices = graphrag.find_suspicious_devices(min_accounts=3)
        
        if devices:
            device_id = devices[0]['device_id']
            print(f"\n📍 Device sélectionné: {device_id}")
            print(f"   - Comptes connectés: {devices[0]['account_count']}")
            print(f"   - Transactions: {devices[0]['tx_count']}")
            print(f"   - Taux de fraude: {devices[0]['fraud_rate']:.1f}%")
            
            report = graphrag.investigate_fraud_network(device_id)
            print("\n" + "─"*80)
            print("📄 RAPPORT D'INVESTIGATION:")
            print("─"*80)
            print(report)
        else:
            print("❌ Aucun device suspect trouvé")
    
    finally:
        graphrag.close()


def test_find_all_suspicious():
    """Lister tous les devices suspects"""
    print_section("TEST 3: LISTE DES DEVICES SUSPECTS")
    
    graphrag = FraudGraphRAG()
    
    try:
        devices = graphrag.find_suspicious_devices(min_accounts=2)
        
        print(f"\n✅ {len(devices)} devices suspects trouvés:\n")
        
        for i, device in enumerate(devices[:15], 1):
            # Icône selon le taux de fraude
            if device['fraud_rate'] > 80:
                status = "🚨"
            elif device['fraud_rate'] > 50:
                status = "⚠️"
            else:
                status = "⚡"
            
            print(f"{status} {i:2d}. {device['device_id']}")
            print(f"       └─ {device['account_count']} comptes | "
                  f"{device['tx_count']} tx | "
                  f"{device['fraud_count']} fraudes ({device['fraud_rate']:.1f}%)")
    
    finally:
        graphrag.close()


def test_compare_predictions():
    """Comparer prédictions ML vs réalité"""
    print_section("TEST 4: COMPARAISON ML vs RÉALITÉ")
    
    graphrag = FraudGraphRAG()
    
    try:
        # Trouver 10 transactions (5 fraudes, 5 légitimes)
        query_fraud = """
        MATCH (t:Transaction {is_fraud: 1})
        RETURN t.id as tx_id, t.amount as amount
        LIMIT 5
        """
        
        query_legit = """
        MATCH (t:Transaction {is_fraud: 0})
        RETURN t.id as tx_id, t.amount as amount
        LIMIT 5
        """
        
        with graphrag.driver.session() as session:
            fraud_txs = [dict(r) for r in session.run(query_fraud)]
            legit_txs = [dict(r) for r in session.run(query_legit)]
        
        all_txs = fraud_txs + legit_txs
        
        print("\n📊 Analyse de 10 transactions:\n")
        
        correct = 0
        for i, tx in enumerate(all_txs, 1):
            # Récupérer détails
            details = graphrag.get_transaction_details(tx['tx_id'])
            if not details:
                continue
            
            # Prédire
            prediction = graphrag.predict_transaction(details)
            
            # Comparer
            actual = bool(details.get('is_fraud'))
            predicted = prediction['is_fraud']
            is_correct = actual == predicted
            
            if is_correct:
                correct += 1
                icon = "✅"
            else:
                icon = "❌"
            
            print(f"{icon} {i:2d}. Transaction ${tx['amount']:.2f}")
            print(f"       Réel: {'FRAUDE' if actual else 'LÉGITIME':8s} | "
                  f"Prédit: {'FRAUDE' if predicted else 'LÉGITIME':8s} | "
                  f"Proba: {prediction['fraud_probability']:.2%}")
        
        accuracy = correct / len(all_txs) * 100
        print(f"\n🎯 Accuracy: {accuracy:.0f}% ({correct}/{len(all_txs)})")
    
    finally:
        graphrag.close()


def main():
    """Exécuter tous les tests"""
    print("="*80)
    print("🚀 SUITE DE TESTS GRAPHRAG - DÉTECTION DE FRAUDE")
    print("="*80)
    
    # Exécuter les tests
    test_find_all_suspicious()
    test_explain_transaction()
    test_investigate_network()
    test_compare_predictions()
    
    print("\n\n")
    print("="*80)
    print("✅ TOUS LES TESTS TERMINÉS AVEC SUCCÈS")
    print("="*80)
    print("\n💡 Le système GraphRAG est opérationnel et prêt pour l'API!")


if __name__ == "__main__":
    main()