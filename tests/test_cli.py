from click.testing import CliRunner
from airone.cli.main import cli
import os

def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ['--help'])
    assert result.exit_code == 0
    assert 'AirOne: Intelligent Semantic Compression Platform.' in result.output

def test_compress_decompress(tmp_path):
    runner = CliRunner()
    
    # Create test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("Hello World" * 100)
    
    # Compress
    result = runner.invoke(cli, ['compress', str(test_file)])
    assert result.exit_code == 0
    assert 'Compression Complete' in result.output
    
    compressed_file = str(test_file) + '.air'
    assert os.path.exists(compressed_file)
    
    # Decompress
    out_file = tmp_path / "out.txt"
    result = runner.invoke(cli, ['decompress', compressed_file, '-o', str(out_file)])
    assert result.exit_code == 0
    assert 'Decompression Complete' in result.output
    
    assert out_file.read_text() == "Hello World" * 100
