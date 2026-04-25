"""
CORTEX server entry point.
"""
import os, sys
sys.path.insert(0, "/app")

from openenv.core.env_server.http_server import create_app
from openenv.core import Environment
from models import CortexAction, CortexObservation, CortexState
from server.cortex_environment import CortexEnvironment

app = create_app(
    env=CortexEnvironment,
    action_cls=CortexAction,
    observation_cls=CortexObservation,
    env_name="cortex",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)