"""Pytest configuration and shared fixtures for ASH test suite."""

import pytest
import numpy as np
import warnings
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for testing


@pytest.fixture(autouse=True, scope="session")
def _configure_test_session():
    """
    Configurazione di sessione per i test:
    - Forza backend matplotlib Agg (headless)
    - Sopprime warning benigni noti dalle routine di viz
    """
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
    except Exception:
        # Non bloccare la suite se matplotlib non è presente
        pass

    # Sopprimi warning noti: legend senza artists nelle figure di test
    warnings.filterwarnings(
        "ignore",
        message=r"No artists with labels found to put in legend\.",
        category=UserWarning,
    )


@pytest.fixture()
def rng() -> np.random.Generator:
    """Generatore deterministico per test che richiedono random."""
    return np.random.default_rng(0)


@pytest.fixture()
def ash_factory():
    """Factory per creare istanze ASH con backend opzionale."""
    from ash_model import ASH

    def _make(backend: str | None = None) -> ASH:
        return ASH() if backend is None else ASH(backend=backend)

    return _make


@pytest.fixture()
def small_hg(ash_factory):
    """Piccolo ipergrafo statico di utilità."""
    a = ash_factory()
    a.add_hyperedge([1, 2, 3], 0)
    a.add_hyperedge([15, 25], 0)
    a.add_hyperedge([1, 24, 34], 0)
    a.add_hyperedge([1, 2, 5, 6], 0)
    a.add_hyperedge([1, 2, 5], 1)
    a.add_hyperedge([3, 4, 5, 10], 1)
    a.add_hyperedge([3, 4, 5, 12], 1)
    return a


@pytest.fixture()
def small_temporal_hg(ash_factory):
    """Piccolo ipergrafo temporale multi-snapshot."""
    a = ash_factory()
    # e1: 1-2 attivo 0..2
    a.add_hyperedge([1, 2], 0, 2)
    # e2: 2-3 attivo 1..3
    a.add_hyperedge([2, 3], 1, 3)
    # e3: 3-4 attivo 3..3
    a.add_hyperedge([3, 4], 3, 3)
    return a


@pytest.fixture()
def matplotlib_agg():
    """Ensure matplotlib uses non-interactive backend for tests."""
    import matplotlib.pyplot as plt

    yield
    # Cleanup
    plt.close("all")
