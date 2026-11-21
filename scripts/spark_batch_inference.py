#!/usr/bin/env python3
"""
scripts/spark_batch_inference.py
================================
Self-contained PySpark mapPartitions batch inference script for SANADA.
Runs distributed multi-task machine listening across campus cluster nodes.

Usage on Spark Cluster:
    spark-submit \
        --master yarn \
        --deploy-mode cluster \
        --num-executors 8 \
        --executor-cores 4 \
        --executor-memory 8G \
        scripts/spark_batch_inference.py \
        --input hdfs:///data/sanada/audio_files.txt \
        --output hdfs:///data/sanada/predictions/ \
        --checkpoint-dir /opt/sanada/models/checkpoints
"""

import os
import sys
import json
import argparse
import logging
from typing import Iterator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SANADA-SparkBatch")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


def run_partition_inference(
    partition_records: Iterator[str],
    checkpoint_dir: str,
    task: str = "all",
    mode: str = "auto"
) -> Iterator[str]:
    """
    Worker function executed on each Spark executor partition.
    Initializes SANADAPipeline ONCE per partition, then processes all audio records.
    """
    import soundfile as sf
    from sanada_inference import SANADAPipeline

    # Enforce offline execution on Spark worker nodes
    os.environ["OFFLINE_MODE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HOME"] = os.path.join(checkpoint_dir, "hf_cache")

    try:
        pipeline = SANADAPipeline(checkpoint_dir=checkpoint_dir, device="cpu")
    except Exception as e:
        logger.error(f"Executor failed to initialize SANADAPipeline: {e}")
        for path in partition_records:
            cleaned = path.strip()
            if cleaned:
                yield json.dumps({
                    "audio_path": cleaned,
                    "status": "error",
                    "error": f"Pipeline initialization failure: {str(e)}"
                })
        return

    for record in partition_records:
        path = record.strip()
        if not path:
            continue
        try:
            data, sr = sf.read(path)
            res = pipeline.run_full_inference(
                waveform_data=data,
                sample_rate=sr,
                task=task,
                filename=os.path.basename(path),
                mode=mode
            )
            output = {
                "audio_path": path,
                "filename": os.path.basename(path),
                "status": "success",
                "predicted_k": res.get("predicted_k"),
                "raw_k": res.get("raw_k"),
                "calibrated_k": res.get("calibrated_k"),
                "human_speech_detected": res.get("human_speech_detected", False),
                "human_speech_prob": res.get("human_speech_prob", 0.0),
                "complexity_ceiling_exceeded": res.get("complexity_ceiling_exceeded", False),
                "active_events": res.get("active_events", []),
                "top_5_events": res.get("top_5_events", []),
                "active_separator": res.get("active_separator"),
                "model_tier": res.get("model_tier", "legacy_fallback"),
                "duration_sec": res.get("duration_sec")
            }
            yield json.dumps(output)
        except Exception as err:
            yield json.dumps({
                "audio_path": path,
                "filename": os.path.basename(path) if path else "",
                "status": "error",
                "error": str(err)
            })


def main():
    parser = argparse.ArgumentParser(description="SANADA Distributed PySpark Batch Inference")
    parser.add_argument("--input", "-i", required=True, help="Input text file containing audio paths or HDFS directory")
    parser.add_argument("--output", "-o", required=True, help="Output directory on HDFS or local filesystem")
    parser.add_argument("--checkpoint-dir", default=os.path.join(BASE_DIR, "models", "checkpoints"), help="Path to checkpoint directory")
    parser.add_argument("--task", default="all", choices=["all", "counting", "tagging", "separation"], help="Task to execute")
    parser.add_argument("--mode", default="auto", choices=["auto", "sanada", "computer", "live_mic"], help="Conditioning mode")
    parser.add_argument("--partitions", type=int, default=0, help="Optional repartition count")
    args = parser.parse_args()

    try:
        from pyspark.sql import SparkSession
    except ImportError:
        logger.error("pyspark is not installed in the current environment.")
        logger.info("Submit this script using spark-submit on the campus cluster.")
        sys.exit(1)

    spark = SparkSession.builder \
        .appName("SANADA-SparkBatchInference") \
        .getOrCreate()

    sc = spark.sparkContext
    logger.info(f"Loaded SparkSession: {spark.version} | Master: {sc.master}")

    input_rdd = sc.textFile(args.input)
    if args.partitions > 0:
        input_rdd = input_rdd.repartition(args.partitions)

    ckpt_dir = os.path.abspath(args.checkpoint_dir)
    task = args.task
    mode = args.mode

    logger.info(f"Starting mapPartitions batch inference from {args.input} -> {args.output}")
    results_rdd = input_rdd.mapPartitions(
        lambda partition: run_partition_inference(partition, checkpoint_dir=ckpt_dir, task=task, mode=mode)
    )

    results_rdd.saveAsTextFile(args.output)
    logger.info(f"Batch inference complete. Results saved to {args.output}")
    spark.stop()


if __name__ == "__main__":
    main()
