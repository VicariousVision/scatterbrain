import asyncio
import logging
from backend.main import lifespan, app
from backend.routers.chat import _chat_service

logging.basicConfig(level=logging.ERROR)

async def test():
    async with lifespan(app):
        print(await _chat_service.query('hello', []))

if __name__ == "__main__":
    asyncio.run(test())
