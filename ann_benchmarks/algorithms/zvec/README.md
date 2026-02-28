# Zvec Integration for ANN-Benchmarks

This directory contains the integration of [Zvec](https://github.com/alibaba/zvec) into the ANN-Benchmarks framework.

## About Zvec

Zvec is an open-source, in-process vector database from Alibaba - lightweight, lightning-fast, and designed to embed directly into applications. Built on Proxima (Alibaba's battle-tested vector search engine), it delivers production-grade, low-latency, scalable similarity search.

**Key Features:**
- Blazing fast: Searches billions of vectors in milliseconds
- Simple installation: `pip install zvec`
- Production-proven performance validated in VectorDBBench
- HNSW graph-based index for optimal speed/accuracy trade-off
- In-process library (no server required)

## Implementation

This integration uses **HNSW (Hierarchical Navigable Small World)** with optimized parameters based on [VectorDBBench](https://github.com/zilliztech/VectorDBBench)'s proven configuration:

**Optimized Parameters:**
- `M=50`: Number of bi-directional links per element
- `efConstruction=500`: Dynamic candidate list size during index construction
- `ef` (query-time): Configurable search quality parameter (10-800)

These parameters have been validated across multiple benchmarks and production deployments to provide the best balance of recall, speed, and memory usage.

## Usage

### Building the Docker Image

```bash
python install.py --algorithm zvec
```

### Running Benchmarks

Run benchmarks on a specific dataset:

```bash
python run.py --algorithm zvec --dataset glove-100-angular
```

### Plotting Results

```bash
python plot.py --dataset glove-100-angular
```

## Configuration

The configuration is defined in [`config.yml`](./config.yml):

- **Index Type**: HNSW only (optimal for ANN benchmarks)
- **Metrics**: Supports both `angular` (cosine) and `euclidean` (L2)
- **Build Parameters**: M=50, efConstruction=500 (VectorDBBench optimized)
- **Query Parameters**: ef sweep from 10 to 800 for recall/QPS trade-off analysis

## Testing

A simple integration test is available:

```bash
python test_zvec_integration.py
```

This verifies that Zvec can build an HNSW index and perform queries correctly.

## Files

- [`module.py`](./module.py): Python wrapper implementing the BaseANN interface
- [`config.yml`](./config.yml): Algorithm configuration with VectorDBBench-optimized parameters
- [`Dockerfile`](./Dockerfile): Docker image definition for reproducible builds
- [`README.md`](./README.md): This documentation

## Performance Notes

This implementation follows best practices from VectorDBBench:
- Memory-mapped I/O enabled for better performance
- Post-insertion optimization for improved query speed
- Optimal HNSW parameters validated across production workloads

## Requirements

- Python 3.10-3.12
- Linux (x86_64, ARM64) or macOS (ARM64)
- zvec package (installed via pip)

## References

- [Zvec GitHub Repository](https://github.com/alibaba/zvec)
- [Zvec Documentation](https://zvec.org/)
- [VectorDBBench](https://github.com/zilliztech/VectorDBBench) - Proven configuration source
- [VectorDBBench Leaderboard](https://zilliz.com/benchmark)

## License

Zvec is licensed under Apache 2.0. This integration follows the ANN-Benchmarks license.
