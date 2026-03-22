from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Backend settings
    HOST: str = "0.0.0.0"
    PORT: int = 80

    # MongoDB (direct connection)
    MONGO_URI: str = "mongodb://mongo_db:27017/"
    DATABASE_NAME: str = "camera_traking"

    # Service URLs (for proxying to AI services)
    DETECTION_SERVICE_URL: str = "http://detection_service:5000"

    # Static files (shared volume)
    STATIC_DIR: str = "/static_files"

    # JWT settings
    SECRET_KEY: str = "your-secret-key"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    class Config:
        env_file = ".env"


settings = Settings()
