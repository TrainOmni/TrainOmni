"""Model plugin contract checks that do not depend on a training engine."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from trainomni.contracts import ValidationIssue, ValidationReport

from .protocol import MODEL_PLUGIN_API_VERSION, ModelPluginManifest

REQUIRED_PLUGIN_METHODS = (
    "capabilities",
    "build",
    "component_catalog",
    "validate_sample",
    "encode",
    "collate",
    "export",
)


def validate_plugin_shape(plugin: Any) -> ValidationReport:
    issues: list[ValidationIssue] = []
    manifest = getattr(plugin, "manifest", None)
    if not isinstance(manifest, ModelPluginManifest):
        issues.append(
            ValidationIssue(
                code="plugin.manifest",
                message="plugin.manifest must be a ModelPluginManifest",
                path="manifest",
            )
        )
    elif manifest.api_version != MODEL_PLUGIN_API_VERSION:
        issues.append(
            ValidationIssue(
                code="plugin.api_version",
                message=(
                    f"plugin API {manifest.api_version!r} is incompatible with "
                    f"{MODEL_PLUGIN_API_VERSION!r}"
                ),
                path="manifest.api_version",
            )
        )

    for method_name in REQUIRED_PLUGIN_METHODS:
        if not callable(getattr(plugin, method_name, None)):
            issues.append(
                ValidationIssue(
                    code="plugin.missing_method",
                    message=f"plugin is missing callable {method_name}()",
                    path=method_name,
                )
            )

    if isinstance(manifest, ModelPluginManifest) and callable(
        getattr(plugin, "capabilities", None)
    ):
        try:
            runtime_capabilities = plugin.capabilities()
        except Exception as exc:  # noqa: BLE001 - isolate untrusted plugin boundary
            issues.append(
                ValidationIssue(
                    code="plugin.capabilities_error",
                    message=f"capabilities() failed: {type(exc).__name__}: {exc}",
                    path="capabilities",
                )
            )
        else:
            if runtime_capabilities != manifest.capabilities:
                issues.append(
                    ValidationIssue(
                        code="plugin.capabilities_mismatch",
                        message="capabilities() differs from manifest.capabilities",
                        path="capabilities",
                    )
                )
    return ValidationReport(tuple(issues))


def validate_plugin_components(
    plugin: Any, bundle: Any, parameter_names: Iterable[str]
) -> ValidationReport:
    issues: list[ValidationIssue] = []
    try:
        catalog = plugin.component_catalog(bundle)
        _, component_issues = catalog.classify(parameter_names)
    except Exception as exc:  # noqa: BLE001 - isolate untrusted plugin boundary
        issues.append(
            ValidationIssue(
                code="plugin.component_error",
                message=f"component audit failed: {type(exc).__name__}: {exc}",
                path="component_catalog",
            )
        )
    else:
        issues.extend(
            ValidationIssue(code=item.code, message=item.message)
            for item in component_issues
        )
    return ValidationReport(tuple(issues))
