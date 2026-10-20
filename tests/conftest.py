from datetime import timedelta

from hypothesis import settings

settings.register_profile(
    "deterministic",
    derandomize=True,
    deadline=timedelta(seconds=5),
)
settings.load_profile("deterministic")
