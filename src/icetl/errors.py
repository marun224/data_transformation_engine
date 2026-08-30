"""Exception hierarchy mirroring `pyspark.errors`.

Drop-in compatibility includes *failure* modes: a script with
`except AnalysisException:` around a missing-table lookup must keep working. So the
public names, their inheritance, and the `getMessage()`/`getErrorClass()` accessors
follow PySpark rather than inventing our own scheme.

icetl-specific errors subclass the PySpark exception a user would expect to catch:
`TableNotFoundError` is an `AnalysisException`, `UnsupportedFeatureError` is a
`PySparkNotImplementedError`, and so on.
"""

from __future__ import annotations

__all__ = [
    "AnalysisException",
    "CatalogNotFoundError",
    "ConfigurationError",
    "IcetlError",
    "IllegalArgumentException",
    "NamespaceNotFoundError",
    "ParseException",
    "PySparkAttributeError",
    "PySparkException",
    "PySparkNotImplementedError",
    "PySparkTypeError",
    "PySparkValueError",
    "PythonException",
    "QueryExecutionException",
    "TableAlreadyExistsException",
    "TableNotFoundError",
    "TempTableAlreadyExistsException",
    "UnsupportedFeatureError",
    "UnsupportedOperationException",
]


class IcetlError(Exception):
    """Root of every error icetl raises. Not part of the PySpark surface."""


class PySparkException(IcetlError):
    """Base class for the PySpark-compatible exceptions.

    Mirrors `pyspark.errors.PySparkException`, including the error-class accessors
    added in Spark 3.4 that user code sometimes branches on.
    """

    def __init__(
        self,
        message: str | None = None,
        error_class: str | None = None,
        message_parameters: dict[str, str] | None = None,
    ) -> None:
        self._message = message or ""
        self._error_class = error_class
        self._message_parameters = message_parameters or {}
        super().__init__(self._message)

    def getMessage(self) -> str:
        if self._error_class is None:
            return self._message
        return f"[{self._error_class}] {self._message}"

    def getErrorClass(self) -> str | None:
        return self._error_class

    def getMessageParameters(self) -> dict[str, str]:
        return dict(self._message_parameters)

    def getSqlState(self) -> str | None:
        """Always None. We do not model SQLSTATE codes; documented in divergence.md."""
        return None


class AnalysisException(PySparkException):
    """Raised when a plan cannot be analysed: unknown table, column, or bad types.

    Raised at *transformation* time, not action time, matching Spark: an unresolvable
    reference must fail at `df.select("typo")`, not later at `df.collect()`.
    """


class ParseException(AnalysisException):
    """Raised when SQL text cannot be parsed."""


class IllegalArgumentException(PySparkException):
    """Raised when an argument is invalid, matching Spark's Java-derived name."""


class UnsupportedOperationException(PySparkException):
    """Raised when an operation is valid Spark but not supported here."""


class QueryExecutionException(PySparkException):
    """Raised when a query fails during execution rather than analysis."""


class PythonException(PySparkException):
    """Raised when user Python code (a UDF) fails."""


class TempTableAlreadyExistsException(AnalysisException):
    """Raised when creating a temp view that already exists."""


class TableAlreadyExistsException(AnalysisException):
    """Raised when creating a table that already exists."""


class PySparkValueError(PySparkException, ValueError):
    """A ValueError that is also catchable as a PySparkException."""


class PySparkTypeError(PySparkException, TypeError):
    """A TypeError that is also catchable as a PySparkException."""


class PySparkAttributeError(PySparkException, AttributeError):
    """An AttributeError that is also catchable as a PySparkException."""


class PySparkNotImplementedError(PySparkException, NotImplementedError):
    """A NotImplementedError that is also catchable as a PySparkException."""


# ---------------------------------------------------------------------------
# icetl-specific errors, each subclassing the PySpark error a user would catch
# ---------------------------------------------------------------------------


class ConfigurationError(IcetlError):
    """Raised when configuration is missing or contradictory.

    Deliberately *not* a PySparkException: it means icetl is set up wrong, which is
    not a condition any Spark script would have a handler for.
    """


class CatalogNotFoundError(AnalysisException):
    """Raised when a table reference names a catalog that is not configured."""


class NamespaceNotFoundError(AnalysisException):
    """Raised when a namespace does not exist in the catalog."""


class TableNotFoundError(AnalysisException):
    """Raised when a table does not exist in the catalog."""


class UnsupportedFeatureError(PySparkNotImplementedError):
    """Raised for a real Spark feature icetl has not implemented yet.

    Carries the phase from PLAN.md that will deliver it, so the error tells the
    reader *when* rather than just *no*.
    """

    def __init__(self, feature: str, phase: str | None = None, hint: str | None = None) -> None:
        self.feature = feature
        self.phase = phase
        self.hint = hint
        detail = f" It is scheduled for {phase}." if phase else ""
        advice = f" {hint.rstrip('.')}." if hint else ""
        super().__init__(
            f"{feature} is not implemented yet.{detail}{advice}",
            error_class="UNSUPPORTED_FEATURE",
            message_parameters={
                "feature": feature,
                **({"phase": phase} if phase else {}),
                **({"hint": hint} if hint else {}),
            },
        )
