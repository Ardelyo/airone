"""
AirOne CLI - Main Entry Point
"""

from __future__ import annotations

import click
from airone.__version__ import __version__
from airone.cli.commands.compress import compress_command
from airone.cli.commands.decompress import decompress_command
from airone.cli.commands.analyze import analyze_command


@click.group()
@click.version_option(version=__version__)
def cli():
    """AirOne: Intelligent Semantic Compression Platform."""
    pass


cli.add_command(compress_command)
cli.add_command(decompress_command)
cli.add_command(analyze_command)


if __name__ == "__main__":
    cli()
