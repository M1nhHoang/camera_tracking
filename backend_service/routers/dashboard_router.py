from fastapi import APIRouter, Depends
from services.dashboard_service import DashboardService

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])

@router.get("/summary")
async def get_dashboard_summary(service: DashboardService = Depends(DashboardService)):
    """Get dashboard summary statistics"""
    return service.get_summary()

@router.get("/detection-activity")
async def get_detection_activity(service: DashboardService = Depends(DashboardService)):
    """Get detection activity data for chart"""
    return service.get_detection_activity()

@router.get("/camera-distribution") 
async def get_camera_distribution(service: DashboardService = Depends(DashboardService)):
    """Get camera distribution statistics"""
    return service.get_camera_distribution()

@router.get("/recent-activity")
async def get_recent_activity(service: DashboardService = Depends(DashboardService)):
    """Get recent system activity"""
    return service.get_recent_activity()

@router.get("/alerts")
async def get_system_alerts(service: DashboardService = Depends(DashboardService)):
    """Get system alerts"""
    return service.get_system_alerts()