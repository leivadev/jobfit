from slowapi import Limiter

from backend.config import resolve_rate_limit_key_func

# In-process storage: breaks under multiple worker processes/instances; known
# limitation, deferred to Phase 6/7. moving-window (not the fixed-window
# default) so the limit is a true rolling minute, not clock-aligned.
limiter = Limiter(key_func=resolve_rate_limit_key_func(), strategy="moving-window")

RECOMMEND_RATE_LIMIT_PER_MINUTE = 3
RECOMMEND_RATE_LIMIT = f"{RECOMMEND_RATE_LIMIT_PER_MINUTE}/minute"
