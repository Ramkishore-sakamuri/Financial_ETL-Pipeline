# Financial Transaction ETL Pipeline

This project implements an end-to-end ETL (Extract, Transform, Load) pipeline designed for processing financial transaction data. It incorporates best practices aimed at improving data processing efficiency and load times.

**Note on Performance Improvement (e.g., "40% faster data load times"):**
The design choices in this pipeline, such as batch processing, efficient data formats (Apache Parquet), and optimized database loading techniques, are geared towards significant performance gains compared to naive row-by-row processing or unoptimized scripts. A specific figure like 40% is achievable when migrating from a less optimized system. To quantify this, you would typically:
1. Benchmark the existing (old) pipeline's data load time.
2. Implement this optimized pipeline.
3. Benchmark the new pipeline's data load time under similar conditions (data volume, hardware).
The improvement = `((OldTime - NewTime) / OldTime) * 100%`.

## Features

* **Modular Design:** Separate scripts for Extract, Transform, and Load stages.
* **Configuration Driven:** Key parameters managed via a YAML configuration file.
* **Performance Optimized:**
    * Chunking for handling large datasets.
    * Use of efficient intermediate data formats (Apache Parquet).
    * Optimized database loading (e.g., Pandas `to_sql` with `method='multi'` or `psycopg2.extras.execute_values`).
* **Basic Logging and Error Handling.**
* **Scalability Considerations:** Designed to be extensible for more complex scenarios or larger data volumes.

## Directory Structure
