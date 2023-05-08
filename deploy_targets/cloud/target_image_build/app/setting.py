from pydantic import BaseSettings

from app.config import read_from_file


class Settings(BaseSettings):
    # config, _ = read_from_file(None, daemon_name='forklift')
    # SECRET_KEY: str = config['auth']['FORKLIFT_SECRET_KEY']
    # # 60 minutes * 24 hours * 8 days = 8 days
    # ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8
    # # BACKEND_CORS_ORIGINS is a JSON-formatted list of origins
    # # e.g: '["http://localhost", "http://localhost:4200", "http://localhost:3000", \
    # # "http://localhost:8080", "http://local.dockertoolbox.tiangolo.com"]'
    # BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = []
    # SQLALCHEMY_DATABASE_URI: Optional[PostgresDsn] = None
    REDIS_HOST = "localhost"  # for development machine
    # REDIS_HOST = 'redis'  # for docker container.
    REDIS_PORT = 6379
    # XREAD_TIMEOUT = 0
    # XREAD_COUNT = 100
    # NUM_PREVIOUS = 30
    STREAM_MAX_LEN = 10000
    DEPLOY_SERVER_URL = "http://127.0.0.1:8890"
    # PORT = 9080
    # HOST = "0.0.0.0"
    SUPPORTED_PRESETS = {
        "image": [
            "yolov5",
        ]
    }

    class Config:
        case_sensitive = True


settings = Settings()
