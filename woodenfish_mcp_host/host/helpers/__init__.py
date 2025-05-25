from datetime import UTC, datetime


def today_datetime() -> str:
    """当前日期和时间。"""
    return datetime.now(tz=UTC).isoformat()
