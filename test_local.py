"""
Quick local test for CORTEX — runs without a live server.
Verifies models and environment logic work correctly.
"""
import asyncio
from server.cortex_environment import CortexEnvironment
from models import CortexAction


async def main():
    env = CortexEnvironment()

    print("=" * 60)
    print("CORTEX LOCAL TEST")
    print("=" * 60)

    # Reset
    obs = await env.reset(difficulty="easy")
    print(f"\n[RESET] Episode started | Difficulty: easy")
    print(f"  Broken pipelines : {obs.broken_pipeline_count}")
    print(f"  HR conflicts     : {len(obs.hr_conflicts)}")
    print(f"  Customer tickets : {len(obs.tickets)}")
    print(f"  Hint             : {obs.tool_hint}")

    # Step 1 — inspect
    result = await env.step(CortexAction(tool_name="inspect_systems", arguments={}))
    print(f"\n[STEP 1] inspect_systems | reward={result.reward:.3f}")
    print(f"  {result.observation.last_action_result[:200]}")

    # Step 2 — repair first broken pipeline
    if obs.pipelines:
        pid = obs.pipelines[0].pipeline_id
        result = await env.step(CortexAction(
            tool_name="repair_pipeline",
            arguments={"pipeline_id": pid}
        ))
        print(f"\n[STEP 2] repair_pipeline({pid}) | reward={result.reward:.3f}")
        print(f"  {result.observation.last_action_result[:200]}")

    # Step 3 — resolve first HR conflict
    if obs.hr_conflicts:
        cid = obs.hr_conflicts[0].conflict_id
        result = await env.step(CortexAction(
            tool_name="resolve_hr",
            arguments={"conflict_id": cid, "decision": "mediate"}
        ))
        print(f"\n[STEP 3] resolve_hr({cid}) | reward={result.reward:.3f}")
        print(f"  {result.observation.last_action_result[:200]}")

    # Step 4 — handle first ticket
    if obs.tickets:
        tid = obs.tickets[0].ticket_id
        result = await env.step(CortexAction(
            tool_name="handle_ticket",
            arguments={"ticket_id": tid, "resolution": "Investigated and fixed root cause."}
        ))
        print(f"\n[STEP 4] handle_ticket({tid}) | reward={result.reward:.3f}")
        print(f"  {result.observation.last_action_result[:200]}")

    print(f"\n{'=' * 60}")
    print(f"Cumulative reward : {result.observation.cumulative_reward:.4f}")
    print(f"Done              : {result.done}")
    print(f"Steps taken       : {result.observation.step}")
    print("=" * 60)
    print("\n✅ Local test passed!")


if __name__ == "__main__":
    asyncio.run(main())