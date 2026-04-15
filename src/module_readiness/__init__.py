__all__ = [
    "ModuleReadinessState",
    "ModuleReadinessQueryAPI",
    "PipelineConfig",
    "run_pipeline",
]


def __getattr__(name: str):
    if name == "ModuleReadinessQueryAPI":
        from .api import ModuleReadinessQueryAPI

        return ModuleReadinessQueryAPI
    if name == "PipelineConfig":
        from .config import PipelineConfig

        return PipelineConfig
    if name in {"ModuleReadinessState", "run_pipeline"}:
        from .orchestration import ModuleReadinessState, run_pipeline

        return {"ModuleReadinessState": ModuleReadinessState, "run_pipeline": run_pipeline}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
