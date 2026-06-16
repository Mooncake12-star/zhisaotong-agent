import os
import socket

_redis = None
_ENABLED = False
_CHECKED = False


def _check_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        result = s.connect_ex((host, port))
        s.close()
        return result == 0
    except Exception:
        return False


def _init():
    global _redis, _ENABLED, _CHECKED
    if _CHECKED:
        return
    _CHECKED = True
    import redis
    host = os.getenv("REDIS_HOST", "localhost")
    port = int(os.getenv("REDIS_PORT", 6379))
    if not _check_port_open(host, port):
        print("[Cache] Redis 未启动，缓存已降级（跳过）")
        return
    try:
        _redis = redis.Redis(host=host, port=port, socket_connect_timeout=2, decode_responses=True)
        _redis.ping()
        _ENABLED = True
        print("[Cache] Redis 已连接")
    except Exception:
        print("[Cache] Redis 连接失败，缓存已降级（跳过）")


def get_cache(key: str) -> str | None:
    _init()
    if not _ENABLED or _redis is None:
        return None
    try:
        return _redis.get(key)
    except Exception:
        return None


def set_cache(key: str, value: str, ttl: int = 300):
    _init()
    if not _ENABLED or _redis is None:
        return
    try:
        _redis.setex(key, ttl, value)
    except Exception:
        pass


def delete_cache(key: str):
    _init()
    if not _ENABLED or _redis is None:
        return
    try:
        _redis.delete(key)
    except Exception:
        pass


def is_enabled() -> bool:
    if not _CHECKED:
        _init()
    return _ENABLED
