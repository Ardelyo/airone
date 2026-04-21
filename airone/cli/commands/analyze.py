"""
AirOne CLI - analyze command
Provides rich, human-readable analysis output without compressing.
"""

from __future__ import annotations

import json
import os

import click

from airone.analysis.engine import AnalysisEngine


@click.command("analyze")
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format.",
)
@click.option(
    "--full",
    is_flag=True,
    default=False,
    help="Show full block-level entropy breakdown.",
)
def analyze_command(input_file: str, output_format: str, full: bool) -> None:
    """
    Analyse a file and report compression potential without compressing.

    \b
    Examples:
        airone analyze document.pdf
        airone analyze photo.png --format json
        airone analyze data.bin --full
    """
    engine = AnalysisEngine()

    with click.progressbar(
        length=1,
        label="Analysing",
        bar_template="%(label)s  %(bar)s  %(info)s",
        width=40,
    ) as bar:
        try:
            report = engine.analyse(input_file)
        except Exception as exc:
            raise click.ClickException(str(exc))
        bar.update(1)

    if output_format == "json":
        _print_json(report, full)
    else:
        _print_text(report, full)


# ---------------------------------------------------------------------------
# Output renderers
# ---------------------------------------------------------------------------

def _print_text(report, full: bool) -> None:
    click.echo()
    click.secho("  AirOne Analysis Report", fg="cyan", bold=True)
    click.secho("  " + "─" * 44, fg="cyan")

    # Identity
    size_mb = report.file_size / 1024 / 1024
    click.echo(f"  {'File':<20} {report.file_name}")
    click.echo(f"  {'Size':<20} {size_mb:.3f} MB  ({report.file_size:,} bytes)")
    click.echo(f"  {'Format':<20} {report.format.type}  ({report.format.category.value})")
    click.echo(f"  {'Confidence':<20} {report.format.confidence:.0%}")
    if report.format.version:
        click.echo(f"  {'Version':<20} {report.format.version}")

    # Entropy
    click.echo()
    click.secho("  Entropy Analysis", fg="yellow", bold=True)
    click.secho("  " + "─" * 44, fg="yellow")

    e = report.entropy.global_entropy
    e_bar = _entropy_bar(e)
    click.echo(f"  {'Entropy':<20} {e:.4f} bits/byte  {e_bar}")
    click.echo(f"  {'Est. ratio':<20} {report.entropy.compressibility_estimate:.1f}x")

    if full and report.entropy.block_entropies:
        click.echo(f"  {'Min block':<20} {report.entropy.min_block_entropy:.4f}")
        click.echo(f"  {'Max block':<20} {report.entropy.max_block_entropy:.4f}")
        click.echo(f"  {'Mean block':<20} {report.entropy.mean_block_entropy:.4f}")

    label, color = _compressibility_label(e)
    click.secho(f"\n  Compressibility: {label}", fg=color, bold=True)

    # Image classification
    if report.image_classification:
        ic = report.image_classification
        click.echo()
        click.secho("  Image Intelligence", fg="magenta", bold=True)
        click.secho("  " + "─" * 44, fg="magenta")
        click.echo(f"  {'Content type':<20} {ic.content_type.value}  "
                   f"({ic.content_confidence:.0%})")
        click.echo(f"  {'Domain':<20} {ic.domain.value}  "
                   f"({ic.domain_confidence:.0%})")
        click.echo(f"  {'Generation':<20} {ic.generation_method.value}  "
                   f"({ic.generation_confidence:.0%})")

        if ic.features:
            f = ic.features
            click.echo(f"  {'Dimensions':<20} {f.width} × {f.height}")
            click.echo(f"  {'Unique colours':<20} {f.palette_size:,}  "
                       f"({f.unique_color_ratio:.2%} of pixels)")
            click.echo(f"  {'Edge density':<20} {f.edge_density:.4f}")
            click.echo(f"  {'Colour entropy':<20} {f.color_entropy:.4f}")

    # Strategy recommendations
    click.echo()
    click.secho("  Strategy Recommendations", fg="green", bold=True)
    click.secho("  " + "─" * 44, fg="green")

    for i, hint in enumerate(report.strategy_hints, start=1):
        marker = "►" if i == 1 else " "
        click.echo(f"  {marker} {i}. {hint}")

    # Notes
    if report.notes:
        click.echo()
        click.secho("  Notes", fg="white", bold=True)
        for note in report.notes:
            click.echo(f"  • {note}")

    # Timing
    click.echo()
    click.secho(
        f"  Analysis completed in {report.analysis_time * 1000:.1f} ms",
        fg="bright_black",
    )
    click.echo()


def _print_json(report, full: bool) -> None:
    output = {
        "file": {
            "name":     report.file_name,
            "path":     report.file_path,
            "size":     report.file_size,
        },
        "format": {
            "type":       report.format.type,
            "mime":       report.format.mime_type,
            "category":   report.format.category.value,
            "version":    report.format.version,
            "confidence": report.format.confidence,
        },
        "entropy": {
            "global":         round(report.entropy.global_entropy, 4),
            "estimated_ratio": report.entropy.compressibility_estimate,
            "is_random":      report.entropy.is_random,
            "is_highly_compressible": report.entropy.is_highly_compressible,
        },
        "strategy_hints":  report.strategy_hints,
        "notes":           report.notes,
        "analysis_time_ms": round(report.analysis_time * 1000, 2),
    }

    if full and report.entropy.block_entropies:
        output["entropy"]["blocks"] = [
            round(e, 4) for e in report.entropy.block_entropies
        ]

    if report.image_classification:
        ic = report.image_classification
        output["image"] = {
            "content_type":          ic.content_type.value,
            "content_confidence":    round(ic.content_confidence, 4),
            "domain":                ic.domain.value,
            "domain_confidence":     round(ic.domain_confidence, 4),
            "generation_method":     ic.generation_method.value,
            "generation_confidence": round(ic.generation_confidence, 4),
        }
        if ic.features:
            f = ic.features
            output["image"]["features"] = {
                "width":                f.width,
                "height":               f.height,
                "unique_color_ratio":   round(f.unique_color_ratio, 4),
                "palette_size":         f.palette_size,
                "edge_density":         round(f.edge_density, 4),
                "color_entropy":        round(f.color_entropy, 4),
                "spatial_uniformity":   round(f.spatial_uniformity, 4),
                "has_transparency":     f.has_transparency,
                "is_grayscale":         f.is_grayscale,
            }

    click.echo(json.dumps(output, indent=2))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entropy_bar(entropy: float, width: int = 20) -> str:
    """Visual bar showing entropy level 0-8."""
    filled = round((entropy / 8.0) * width)
    bar    = "█" * filled + "░" * (width - filled)
    return f"[{bar}]"


def _compressibility_label(entropy: float) -> tuple[str, str]:
    if entropy < 2.0:
        return "Excellent  (>10x expected)", "green"
    if entropy < 4.0:
        return "Good       (4-10x expected)", "green"
    if entropy < 6.0:
        return "Moderate   (2-4x expected)", "yellow"
    if entropy < 7.5:
        return "Limited    (1-2x expected)", "yellow"
    return "Poor       (~1x, nearly incompressible)", "red"
