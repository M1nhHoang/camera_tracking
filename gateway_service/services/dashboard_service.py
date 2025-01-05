from datetime import datetime, timedelta
import random

class DashboardService:
    def get_summary(self):
        """Get dashboard summary statistics"""
        return {
            "total_users": random.randint(100, 500),
            "new_users": random.randint(5, 20),
            "active_cameras": random.randint(10, 30),
            "total_cameras": random.randint(30, 50),
            "today_detections": random.randint(1000, 5000),
            "detection_change": random.randint(-10, 10)
        }

    def get_detection_activity(self):
        """Get detection activity for the chart"""
        # Generate hourly data for the last 24 hours
        labels = []
        values = []
        now = datetime.now()
        
        for i in range(24):
            hour = now - timedelta(hours=i)
            labels.insert(0, hour.strftime("%H:00"))
            values.insert(0, random.randint(50, 200))

        return {
            "labels": labels,
            "values": values
        }

    def get_camera_distribution(self):
        """Get camera distribution statistics"""
        # Generate reasonable numbers that add up to total_cameras
        total = 30  # Total number of cameras
        active = random.randint(15, 20)
        inactive = random.randint(5, 8)
        error = random.randint(1, 3)
        maintenance = total - active - inactive - error

        return {
            "labels": ["Active", "Inactive", "Error", "Maintenance"],
            "values": [active, inactive, error, maintenance]
        }

    def get_recent_activity(self):
        """Get recent system activity"""
        activities = []
        events = [
            "User Detection",
            "Camera Started",
            "Camera Stopped",
            "System Update",
            "New User Added"
        ]
        locations = ["Main Entrance", "Lobby", "Parking", "Office Area", "Cafeteria"]
        statuses = ["success", "error"]

        for _ in range(10):
            time_offset = random.randint(1, 60)
            activity_time = datetime.now() - timedelta(minutes=time_offset)
            
            activities.append({
                "time": activity_time.strftime("%H:%M:%S"),
                "event": random.choice(events),
                "location": random.choice(locations),
                "status": random.choice(statuses)
            })

        return sorted(activities, key=lambda x: x["time"], reverse=True)

    def get_system_alerts(self):
        """Get system alerts"""
        alerts = []
        alert_types = [
            {
                "level": "error",
                "title": "Camera Offline",
                "message": "Camera in Main Entrance is not responding"
            },
            {
                "level": "warning",
                "title": "High CPU Usage",
                "message": "System CPU usage above 80%"
            },
            {
                "level": "info",
                "title": "System Update",
                "message": "New system update available"
            }
        ]

        # Return 2-4 random alerts
        num_alerts = random.randint(2, 4)
        return random.sample(alert_types, num_alerts)