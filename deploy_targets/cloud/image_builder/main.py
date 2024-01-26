import uvicorn

from app.config import read_from_file
from app.utils import create_logger

if __name__ == "__main__":
    config, _ = read_from_file(None, daemon_name="forklift")
    log = create_logger("image.builder.main")
    log.info(f"config : {config}")
    uvicorn.run(
        "app.server:app",
        host="0.0.0.0",
        port=7007,
    )
