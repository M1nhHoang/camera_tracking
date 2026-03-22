from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API Gateway settings
    GATEWAY_HOST: str = "0.0.0.0"
    GATEWAY_PORT: int = 80

    # Service URLs
    DATABASE_SERVICE_URL: str = "http://database_service:8003"
    RECOGNITION_SERVICE_URL: str = "http://recognition_service:8002"
    FACE_EMBEDDING_SERVICE_URL: str = "http://face_embedding_service:8001"
    DETECTION_SERVICE_URL: str = "http://detection_service:5000"

    # Static files URL for database service
    DATABASE_STATIC_URL: str = "http://database_service:8003/static"

    # JWT settings
    SECRET_KEY: str = "your-secret-key"  # Change in production
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    class Config:
        env_file = ".env"


settings = Settings()
