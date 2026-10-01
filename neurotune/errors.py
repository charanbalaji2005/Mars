"""Exception hierarchy. CLI commands translate these into clear messages and exit codes."""


class NeuroTuneError(Exception):
    """Base class for all expected, user-facing errors."""


class ConfigError(NeuroTuneError):
    """The experiment configuration is invalid. Raised before any GPU work starts."""


class EnvironmentCheckError(NeuroTuneError):
    """A required dependency or device capability is missing."""


class FatalGPUError(NeuroTuneError):
    """The CUDA context is probably corrupted (e.g. illegal memory access).

    Subsequent trials in the same process cannot be trusted, so the experiment
    is stopped and marked as failed rather than continuing silently.
    """


class StoreError(NeuroTuneError):
    """The experiment store is missing, inconsistent, or an experiment cannot be resumed."""
