from band.preprocessing import DefaultPreprocessor
import dataclasses
import re
from datetime import datetime, timezone

# Record the exact moment this agent instance started.
# Any Band message with created_at BEFORE this time is a historical/replayed
# message from startup resync and should be silently ignored.
_AGENT_START_TIME = datetime.now(timezone.utc)


class CustomPreprocessor(DefaultPreprocessor):
    async def process(self, ctx, event, agent_id):
        # Delegate to the default preprocessing logic (resolves sender name, loads history, etc.)
        result = await super().process(ctx, event, agent_id)

        if result is None:
            return None

        # ── Startup resync filter ────────────────────────────────────────────
        # Band replays unprocessed messages on every restart. Skip any message
        # that was created before this process started to avoid re-processing
        # old room history.
        if result.msg and result.msg.created_at:
            if result.msg.created_at < _AGENT_START_TIME:
                return None  # silently drop — old message from before startup

        if result.msg and result.msg.content:
            content = result.msg.content
            # Remove agent self-mentions in formats like @[[uuid]] or @uuid to avoid model confusion
            content = re.sub(rf'@\[\[{agent_id}\]\]', '', content)
            content = re.sub(rf'@{agent_id}', '', content)

            # Since PlatformMessage and AgentInput are frozen dataclasses, we use dataclasses.replace
            new_msg = dataclasses.replace(result.msg, content=content.strip())
            result = dataclasses.replace(result, msg=new_msg)
        return result
