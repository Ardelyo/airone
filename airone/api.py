from .orchestrator.orchestrator import CompressionOrchestrator

class AirOne:
    """
    Main entrypoint for AirOne API.
    """
    def __init__(self):
        self.orchestrator = CompressionOrchestrator()
        
    def compress_file(self, input_path, output_path):
        """Compress a file and save as .air"""
        return self.orchestrator.compress_file(input_path, output_path)

    def decompress_file(self, input_path, output_path):
        """Decompress an .air file back to original"""
        return self.orchestrator.decompress_file(input_path, output_path)
