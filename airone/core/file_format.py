import struct
import json
from ..exceptions import FormatError
from ..compressors.base import CompressionResult

class AirFileFormat:
    """
    Handles .air container format I/O
    Critical for any compression/decompression
    """
    MAGIC = b'AIR1'
    VERSION = 1
    HEADER_SIZE = 512
    
    @staticmethod
    def write(output_path, result: CompressionResult):
        """Write .air file"""
        with open(output_path, 'wb') as f:
            # 1. Prepare header
            # For simplicity in Phase 1, we use a basic header format
            # Magic + Version + Original Size + Compressed Size + Metadata Len
            metadata_bytes = json.dumps(result.metadata).encode('utf-8')
            metadata_len = len(metadata_bytes)
            
            # Pack: 4s H Q Q Q (4 bytes str, 2 bytes unshort, 8 bytes ullong x3)
            # 4 + 2 + 8 + 8 + 8 = 30 bytes
            header = struct.pack('<4sHQQQ', AirFileFormat.MAGIC, AirFileFormat.VERSION, 
                                 result.original_size, len(result.compressed_data), metadata_len)
            
            # Pad header to HEADER_SIZE
            padded_header = header.ljust(AirFileFormat.HEADER_SIZE, b'\0')
            f.write(padded_header)
            
            # 2. Write metadata
            f.write(metadata_bytes)
            
            # 3. Write data
            f.write(result.compressed_data)
    
    @staticmethod
    def read(input_path):
        """Read .air file and return basic data"""
        with open(input_path, 'rb') as f:
            header = f.read(AirFileFormat.HEADER_SIZE)
            if len(header) < AirFileFormat.HEADER_SIZE:
                raise FormatError("File too small to be a valid .air archive")
            
            magic, version, original_size, comp_size, meta_len = struct.unpack('<4sHQQQ', header[:30])
            
            if magic != AirFileFormat.MAGIC:
                raise FormatError(f"Invalid magic bytes. Expected {AirFileFormat.MAGIC}, got {magic}")
            if version != AirFileFormat.VERSION:
                raise FormatError(f"Unsupported AirOne format version: {version}")
                
            metadata_bytes = f.read(meta_len)
            metadata = json.loads(metadata_bytes.decode('utf-8'))
            
            compressed_data = f.read(comp_size)
            
            return {
                "original_size": original_size,
                "compressed_size": comp_size,
                "metadata": metadata,
                "compressed_data": compressed_data
            }

    @staticmethod
    def validate(file_path):
        """Verify file integrity"""
        # Read header logic wrapped around read function for validation
        try:
            AirFileFormat.read(file_path)
            return True
        except FormatError:
            return False
