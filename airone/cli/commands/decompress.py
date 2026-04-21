"""
AirOne CLI - decompress command
"""

from __future__ import annotations

import click
from airone.api import AirOne


@click.command("decompress")
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-o", "--output", help="Output file path")
@click.option("--verbose", is_flag=True, help="Verbose output")
def decompress_command(input_file: str, output: str | None, verbose: bool) -> None:
    """Decompress an .air file."""
    if not input_file.endswith(".air"):
        raise click.ClickException("Input file must have .air extension.")

    if not output:
        output = input_file[:-4]  # Remove .air extension

    airone = AirOne()

    with click.progressbar(length=100, label="Decompressing") as bar:
        try:
            size = airone.decompress_file(input_file, output)
            bar.update(100)
        except Exception as exc:
            raise click.ClickException(str(exc))

    click.echo()
    click.secho("  Decompression Complete", fg="green", bold=True)
    click.echo(f"  {'Output':<15} {output}")
    click.echo(f"  {'Size':<15} {size:,} bytes")
    click.echo()
