# app/health_handler.py
from fastapi import APIRouter, HTTPException
from typing import Dict
import psutil
import os
from datetime import datetime
from sqlalchemy import create_engine, text
from qdrant_client import QdrantClient

# Create a router for health-related endpoints
health_router = APIRouter(prefix="/health", tags=["health"])

class HealthMonitor:
    def __init__(self, postgres_url: str, qdrant_host: str):
        self.postgres_url = postgres_url
        self.qdrant_host = qdrant_host
        self.start_time = datetime.now()
        
        # Initialize database connections
        self.pg_engine = create_engine(postgres_url)
        self.qdrant_client = QdrantClient(host=qdrant_host)

    async def check_postgres(self) -> Dict:
        """Check PostgreSQL connection and basic stats"""
        try:
            with self.pg_engine.connect() as conn:
                # Check connection
                conn.execute(text("SELECT 1"))
                
                # Get database size
                size_result = conn.execute(text(
                    "SELECT pg_database_size(current_database())/1024/1024 as size_mb"
                )).first()
                
                # Get connection count
                conn_result = conn.execute(text(
                    "SELECT count(*) FROM pg_stat_activity"
                )).first()

                return {
                    "status": "healthy",
                    "size_mb": size_result[0],
                    "active_connections": conn_result[0]
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    async def check_qdrant(self) -> Dict:
        """Check Qdrant connection and collection stats"""
        try:
            collections = self.qdrant_client.get_collections()
            collection_stats = {}
            
            for collection in collections.collections:
                info = self.qdrant_client.get_collection(collection.name)
                collection_stats[collection.name] = {
                    "vectors_count": info.vectors_count,
                    "segments_count": info.segments_count
                }

            return {
                "status": "healthy",
                "collections": collection_stats
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    async def check_system(self) -> Dict:
        """Check system resources"""
        return {
            "cpu": {
                "percent": psutil.cpu_percent(interval=1),
                "count": psutil.cpu_count()
            },
            "memory": {
                "total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
                "percent": psutil.virtual_memory().percent
            },
            "disk": {
                "total_gb": round(psutil.disk_usage('/').total / (1024**3), 2),
                "free_gb": round(psutil.disk_usage('/').free / (1024**3), 2),
                "percent": psutil.disk_usage('/').percent
            }
        }

    def get_uptime(self) -> str:
        """Calculate system uptime"""
        uptime = datetime.now() - self.start_time
        return str(uptime).split('.')[0]  # Remove microseconds

# Create health monitor instance
monitor = None

def init_health_monitor(postgres_url: str, qdrant_host: str):
    """Initialize the health monitor"""
    global monitor
    monitor = HealthMonitor(postgres_url, qdrant_host)

# Health check endpoints
@health_router.get("/")
async def health_check() -> Dict:
    """
    Main health check endpoint that returns overall system status
    """
    if not monitor:
        raise HTTPException(status_code=500, detail="Health monitor not initialized")

    pg_status = await monitor.check_postgres()
    qdrant_status = await monitor.check_qdrant()
    system_status = await monitor.check_system()

    return {
        "status": "healthy" if all(s.get("status", "") == "healthy" 
                                 for s in [pg_status, qdrant_status]) else "degraded",
        "uptime": monitor.get_uptime(),
        "databases": {
            "postgres": pg_status,
            "qdrant": qdrant_status
        },
        "system": system_status,
        "timestamp": datetime.now().isoformat()
    }

@health_router.get("/postgres")
async def postgres_health() -> Dict:
    """
    Detailed PostgreSQL health check
    """
    if not monitor:
        raise HTTPException(status_code=500, detail="Health monitor not initialized")
    return await monitor.check_postgres()

@health_router.get("/qdrant")
async def qdrant_health() -> Dict:
    """
    Detailed Qdrant health check
    """
    if not monitor:
        raise HTTPException(status_code=500, detail="Health monitor not initialized")
    return await monitor.check_qdrant()

@health_router.get("/system")
async def system_health() -> Dict:
    """
    Detailed system resource check
    """
    if not monitor:
        raise HTTPException(status_code=500, detail="Health monitor not initialized")
    return await monitor.check_system()