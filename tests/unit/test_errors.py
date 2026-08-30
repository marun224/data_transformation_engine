"""The exception hierarchy.

These assertions look trivial but they are the stability contract: a script with
`except AnalysisException:` around a missing table keeps working only if the
inheritance stays as documented.
"""

from __future__ import annotations

import pytest

from icetl.errors import (
    AnalysisException,
    CatalogNotFoundError,
    ConfigurationError,
    EngineException,
    EngineNotImplementedError,
    EngineValueError,
    IcetlError,
    ParseException,
    TableNotFoundError,
    UnsupportedFeatureError,
)


@pytest.mark.parametrize(
    ("error", "expected_bases"),
    [
        (AnalysisException, (EngineException, IcetlError, Exception)),
        (ParseException, (AnalysisException,)),
        (TableNotFoundError, (AnalysisException,)),
        (CatalogNotFoundError, (AnalysisException,)),
        (UnsupportedFeatureError, (EngineNotImplementedError, NotImplementedError)),
        (EngineValueError, (EngineException, ValueError)),
    ],
)
def test_inheritance_is_catchable_as_expected(
    error: type[Exception], expected_bases: tuple[type[Exception], ...]
) -> None:
    for base in expected_bases:
        assert issubclass(error, base), f"{error.__name__} must be catchable as {base.__name__}"


def test_configuration_error_is_not_an_engine_exception() -> None:
    """Our setup being wrong is not a condition a script would handle."""
    assert issubclass(ConfigurationError, IcetlError)
    assert not issubclass(ConfigurationError, EngineException)


def test_error_class_accessors() -> None:
    error = AnalysisException(
        "boom", error_class="TABLE_OR_VIEW_NOT_FOUND", message_parameters={"t": "x"}
    )
    assert error.getErrorClass() == "TABLE_OR_VIEW_NOT_FOUND"
    assert error.getMessage() == "[TABLE_OR_VIEW_NOT_FOUND] boom"
    assert error.getMessageParameters() == {"t": "x"}
    assert str(error) == "boom"


def test_plain_message_has_no_error_class_prefix() -> None:
    assert AnalysisException("boom").getMessage() == "boom"


def test_unsupported_feature_names_the_phase() -> None:
    """The error should say *when*, not just no."""
    error = UnsupportedFeatureError("MERGE INTO", phase="Phase 8")
    assert "MERGE INTO" in str(error)
    assert "Phase 8" in str(error)
    assert error.getMessageParameters() == {"feature": "MERGE INTO", "phase": "Phase 8"}

    with pytest.raises(NotImplementedError):
        raise error


def test_value_error_is_catchable_both_ways() -> None:
    for catch in (ValueError, EngineException):
        with pytest.raises(catch):
            raise EngineValueError("bad argument")
