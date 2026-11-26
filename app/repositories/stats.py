from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession


async def database_size_bytes(session: AsyncSession) -> int:
    result = await session.execute(select(func.pg_database_size(func.current_database())))
    return int(result.scalar_one())
