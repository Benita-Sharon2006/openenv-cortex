"""
CORTEX client — connects to the running HF Space / local server.
Follows the OpenEnv EnvClient contract exactly.
"""
from __future__ import annotations
from openenv.core import EnvClient
from .models import CortexAction, CortexObservation, CortexState


class CortexEnv(EnvClient):
    """
    Remote client for the CORTEX environment.

    Usage (async):
        async with CortexEnv(base_url="https://<your-space>.hf.space") as env:
            obs = await env.reset(difficulty="hard")
            result = await env.step(CortexAction(
                tool_name="inspect_systems", arguments={}
            ))

    Usage (sync):
        with CortexEnv(base_url="...").sync() as env:
            obs = env.reset()
            result = env.step(CortexAction(tool_name="inspect_systems"))
    """

    action_type = CortexAction
    observation_type = CortexObservation
    state_type = CortexState