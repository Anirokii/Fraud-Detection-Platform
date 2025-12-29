"""
Fraud Detection REST API
=========================
API FastAPI pour exposer le système de détection de fraude

Endpoints:
- POST /api/v1/predict - Prédire une transaction
- POST /api/v1/explain - Expliquer une fraude
- POST /api/v1/investigate - Investiguer un réseau
- GET /api/v1/devices/suspicious - Lister devices suspects
- GET /api/v1/health - Health check
- GET /api/v1/stats - Statistiques du système
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import sys
from pathlib import Path
import warnings
import logging

# Supprimer warnings
warnings.filterwarnings('ignore')
logging.getLogger('neo4j').setLevel(logging.ERROR)

# Ajouter le chemin parent pour importer
sys.path.append(str(Path(__file__).parent.parent))

from graphrag.fraud_graphrag import FraudGraphRAG

# Configuration
app = FastAPI(
    title="Fraud Detection API",
    description="API REST pour la détection et l'explication de fraudes bancaires",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS (permet les requêtes cross-origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instance globale du GraphRAG
graphrag = None


# ========== MODELS PYDANTIC ==========

class TransactionInput(BaseModel):
    """Modèle pour une transaction à analyser"""
    transaction_id: str = Field(..., description="ID de la transaction (ex: TX_xxxxx)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "TX_bada495d847ca4b0527ff6fcd8f3a10d"
            }
        }


class TransactionData(BaseModel):
    """Données complètes d'une transaction pour prédiction"""
    amount: float = Field(..., description="Montant de la transaction")
    hour: Optional[int] = Field(None, description="Heure (0-23)")
    is_weekend: Optional[bool] = Field(None, description="Transaction le weekend")
    is_night: Optional[bool] = Field(None, description="Transaction la nuit")
    velocity: Optional[float] = Field(None, description="Vélocité (tx/heure)")
    amt_deviation: Optional[float] = Field(None, description="Déviation du montant")
    customer_tx_count: Optional[int] = Field(None, description="Nb de transactions du client")
    customer_age: Optional[int] = Field(None, description="Âge du client")
    category: Optional[str] = Field(None, description="Catégorie")
    gender: Optional[str] = Field(None, description="Genre (M/F)")
    state: Optional[str] = Field(None, description="État")
    
    class Config:
        json_schema_extra = {
            "example": {
                "amount": 5000.0,
                "hour": 3,
                "is_weekend": False,
                "is_night": True,
                "velocity": 8.0,
                "amt_deviation": 10.5,
                "customer_tx_count": 150,
                "customer_age": 35,
                "category": "shopping_net",
                "gender": "M",
                "state": "CA"
            }
        }


class PredictionResponse(BaseModel):
    """Réponse de prédiction"""
    is_fraud: bool = Field(..., description="Transaction frauduleuse?")
    fraud_probability: float = Field(..., description="Probabilité de fraude (0-1)")
    risk_level: str = Field(..., description="Niveau de risque (LOW/MEDIUM/HIGH)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "is_fraud": True,
                "fraud_probability": 0.89,
                "risk_level": "HIGH"
            }
        }


class ExplanationResponse(BaseModel):
    """Réponse d'explication"""
    transaction_id: str
    explanation: str = Field(..., description="Explication en langage naturel")
    prediction: PredictionResponse
    device_info: Dict
    
    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "TX_12345",
                "explanation": "Cette transaction est frauduleuse car...",
                "prediction": {
                    "is_fraud": True,
                    "fraud_probability": 0.89,
                    "risk_level": "HIGH"
                },
                "device_info": {
                    "device_id": "DEV_FRAUD_001",
                    "account_count": 5,
                    "fraud_rate": 80.0
                }
            }
        }


class DeviceInput(BaseModel):
    """Modèle pour investiguer un device"""
    device_id: str = Field(..., description="ID du device")
    
    class Config:
        json_schema_extra = {
            "example": {
                "device_id": "DEV_FRAUD_000001"
            }
        }


class InvestigationResponse(BaseModel):
    """Réponse d'investigation"""
    device_id: str
    report: str = Field(..., description="Rapport d'investigation")
    statistics: Dict
    
    class Config:
        json_schema_extra = {
            "example": {
                "device_id": "DEV_FRAUD_001",
                "report": "Investigation du réseau...",
                "statistics": {
                    "account_count": 5,
                    "tx_count": 120,
                    "fraud_count": 95,
                    "fraud_rate": 79.2
                }
            }
        }


class SuspiciousDevice(BaseModel):
    """Device suspect"""
    device_id: str
    account_count: int
    tx_count: int
    fraud_count: int
    fraud_rate: float


class HealthResponse(BaseModel):
    """Statut de santé de l'API"""
    status: str
    neo4j_connected: bool
    llm_loaded: bool
    ml_model_loaded: bool


class StatsResponse(BaseModel):
    """Statistiques du système"""
    total_devices: int
    suspicious_devices: int
    total_accounts: int
    total_transactions: int
    fraud_transactions: int
    fraud_rate: float


# ========== STARTUP / SHUTDOWN ==========

@app.on_event("startup")
async def startup_event():
    """Initialiser le GraphRAG au démarrage"""
    global graphrag
    print("🚀 Initialisation du système GraphRAG...")
    try:
        graphrag = FraudGraphRAG()
        print("✅ GraphRAG initialisé avec succès!")
    except Exception as e:
        print(f"❌ Erreur d'initialisation: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Fermer les connexions"""
    global graphrag
    if graphrag:
        graphrag.close()
        print("🔌 Connexions fermées")


# ========== ENDPOINTS ==========

@app.get("/", tags=["Root"])
async def root():
    """Page d'accueil de l'API"""
    return {
        "message": "Fraud Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Vérifier la santé du système
    
    Returns:
        HealthResponse: Statut de tous les composants
    """
    if not graphrag:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GraphRAG not initialized"
        )
    
    # Tester la connexion Neo4j
    neo4j_ok = False
    try:
        with graphrag.driver.session() as session:
            session.run("RETURN 1")
        neo4j_ok = True
    except:
        pass
    
    return HealthResponse(
        status="healthy" if neo4j_ok else "degraded",
        neo4j_connected=neo4j_ok,
        llm_loaded=graphrag.llm is not None,
        ml_model_loaded=graphrag.ml_model is not None
    )


@app.get("/api/v1/stats", response_model=StatsResponse, tags=["System"])
async def get_stats():
    """
    Obtenir les statistiques globales du système
    
    Returns:
        StatsResponse: Statistiques du graphe
    """
    if not graphrag:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GraphRAG not initialized"
        )
    
    try:
        with graphrag.driver.session() as session:
            # Compter les devices
            result = session.run("MATCH (d:Device) RETURN count(d) as count")
            total_devices = result.single()['count']
            
            # Compter devices suspects
            result = session.run("""
                MATCH (d:Device)<-[:USING]-(t:Transaction)<-[:MADE]-(a:Account)
                WITH d, COUNT(DISTINCT a) as account_count
                WHERE account_count > 1
                RETURN count(d) as count
            """)
            suspicious_devices = result.single()['count']
            
            # Compter accounts
            result = session.run("MATCH (a:Account) RETURN count(a) as count")
            total_accounts = result.single()['count']
            
            # Compter transactions
            result = session.run("""
                MATCH (t:Transaction)
                RETURN count(t) as total,
                       sum(CASE WHEN t.is_fraud = 1 THEN 1 ELSE 0 END) as frauds
            """)
            record = result.single()
            total_tx = record['total']
            fraud_tx = record['frauds']
            fraud_rate = (fraud_tx / total_tx * 100) if total_tx > 0 else 0
        
        return StatsResponse(
            total_devices=total_devices,
            suspicious_devices=suspicious_devices,
            total_accounts=total_accounts,
            total_transactions=total_tx,
            fraud_transactions=fraud_tx,
            fraud_rate=round(fraud_rate, 2)
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting stats: {str(e)}"
        )


@app.post("/api/v1/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_transaction(transaction: TransactionData):
    """
    Prédire si une transaction est frauduleuse
    
    Args:
        transaction: Données de la transaction
    
    Returns:
        PredictionResponse: Prédiction avec probabilité
    """
    if not graphrag:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GraphRAG not initialized"
        )
    
    try:
        # Convertir le modèle Pydantic en dict
        tx_data = transaction.model_dump()
        
        # Faire la prédiction
        prediction = graphrag.predict_transaction(tx_data)
        
        return PredictionResponse(**prediction)
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )


@app.post("/api/v1/explain", response_model=ExplanationResponse, tags=["Explanation"])
async def explain_transaction(transaction: TransactionInput):
    """
    Expliquer pourquoi une transaction est frauduleuse
    
    Args:
        transaction: ID de la transaction
    
    Returns:
        ExplanationResponse: Explication détaillée
    """
    if not graphrag:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GraphRAG not initialized"
        )
    
    try:
        tx_id = transaction.transaction_id
        
        # Vérifier que la transaction existe
        tx_details = graphrag.get_transaction_details(tx_id)
        if not tx_details:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Transaction {tx_id} not found"
            )
        
        # Générer l'explication
        explanation = graphrag.explain_transaction(tx_id)
        
        # Prédiction ML
        prediction = graphrag.predict_transaction(tx_details)
        
        # Info du device
        device_id = tx_details.get('device_id', 'UNKNOWN')
        device_network = graphrag.get_device_network(device_id)
        
        return ExplanationResponse(
            transaction_id=tx_id,
            explanation=explanation,
            prediction=PredictionResponse(**prediction),
            device_info={
                "device_id": device_id,
                "account_count": device_network['account_count'],
                "tx_count": device_network['tx_count'],
                "fraud_count": device_network['fraud_count'],
                "fraud_rate": round(
                    device_network['fraud_count'] / max(device_network['tx_count'], 1) * 100,
                    2
                )
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Explanation error: {str(e)}"
        )


@app.post("/api/v1/investigate", response_model=InvestigationResponse, tags=["Investigation"])
async def investigate_device(device: DeviceInput):
    """
    Investiguer un réseau de fraude autour d'un device
    
    Args:
        device: ID du device
    
    Returns:
        InvestigationResponse: Rapport d'investigation
    """
    if not graphrag:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GraphRAG not initialized"
        )
    
    try:
        device_id = device.device_id
        
        # Récupérer les stats du device
        device_network = graphrag.get_device_network(device_id)
        
        if device_network['account_count'] == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Device {device_id} not found or has no transactions"
            )
        
        # Générer le rapport
        report = graphrag.investigate_fraud_network(device_id)
        
        return InvestigationResponse(
            device_id=device_id,
            report=report,
            statistics={
                "account_count": device_network['account_count'],
                "tx_count": device_network['tx_count'],
                "fraud_count": device_network['fraud_count'],
                "fraud_rate": round(
                    device_network['fraud_count'] / max(device_network['tx_count'], 1) * 100,
                    2
                )
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Investigation error: {str(e)}"
        )


@app.get("/api/v1/devices/suspicious", response_model=List[SuspiciousDevice], tags=["Devices"])
async def get_suspicious_devices(
    min_accounts: int = 2,
    limit: int = 20
):
    """
    Obtenir la liste des devices suspects
    
    Args:
        min_accounts: Nombre minimum de comptes pour être suspect
        limit: Nombre maximum de résultats
    
    Returns:
        List[SuspiciousDevice]: Liste des devices suspects
    """
    if not graphrag:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GraphRAG not initialized"
        )
    
    try:
        devices = graphrag.find_suspicious_devices(min_accounts=min_accounts)
        
        # Limiter les résultats
        devices = devices[:limit]
        
        return [SuspiciousDevice(**device) for device in devices]
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching devices: {str(e)}"
        )


# ========== MAIN ==========

if __name__ == "__main__":
    import uvicorn
    
    print("="*70)
    print("🚀 FRAUD DETECTION API")
    print("="*70)
    print("\n📡 Starting server...")
    print("   - URL: http://localhost:8000")
    print("   - Docs: http://localhost:8000/docs")
    print("   - ReDoc: http://localhost:8000/redoc")
    print("\n" + "="*70)
    
    uvicorn.run(
        "fraud_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )