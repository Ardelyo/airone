"""
AirOne CLI - compress command
"""

from __future__ import annotations

import click
from airone.api import AirOne


@click.command("compress")
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-o", "--output", help="Output file path (.air)")
@click.option("--verbose", is_flag=True, help="Verbose output")
def compress_command(input_file: str, output: str | None, verbose: bool) -> None:
    """Compress a file into an .air container."""
    if not output:
        output = input_file + ".air"

    airone = AirOne()

    with click.progressbar(length=100, label="Compressing") as bar:
        try:
            result = airone.compress_file(input_file, output)
            bar.update(100)
        except Exception as exc:
            raise click.ClickException(str(exc))

    click.echo()
    click.secho("  Compression Complete", fg="green", bold=True)
    click.echo(f"  {'Output':<15} {output}")
    click.echo(f"  {'Original':<15} {result.original_size:,} bytes")
    click.echo(f"  {'Compressed':<15} {result.compressed_size:,} bytes")
    click.echo(f"  {'Ratio':<15} {result.ratio:.2f}x")
    click.echo(f"  {'Strategy':<15} {result.strategy_name}")
    click.echo()
