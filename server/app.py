"""
CORTEX server entry point.
Wires the CortexEnvironment into the OpenEnv FastAPI app.
"""
import os
from openenv.core.env_server.http_server import create_app
from cortex_env.models import CortexAction, CortexObservation, CortexState
from .cortex_environment import CortexEnvironment

app = create_app(
    env_class=CortexEnvironment,
    action_type=CortexAction,
    observation_type=CortexObservation,
    state_type=CortexState,
    enable_web_interface=os.getenv("ENABLE_WEB_INTERFACE", "true").lower() == "true",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)