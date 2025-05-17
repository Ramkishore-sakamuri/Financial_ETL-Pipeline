# src/utils/performance_logger.py

import time
import logging
from typing import Dict, Any, Optional, Union
from functools import wraps

# Get a specific logger for performance metrics.
# This logger should be configured in your logging_config.yaml
# to output to a dedicated performance log file.
perf_logger = logging.getLogger('performance_logger')

class PerformanceLogger:
    """
    A utility class to log performance metrics, primarily execution time, for tasks.
    """

    def __init__(self, run_id: Optional[str] = None, default_context: Optional[Dict[str, Any]] = None):
        """
        Initializes the PerformanceLogger.

        Args:
            run_id (Optional[str]): An optional identifier for the current ETL run.
            default_context (Optional[Dict[str, Any]]): Default key-value pairs to include in every log message.
        """
        self._start_times: Dict[str, float] = {}
        self.run_id = run_id
        self.default_context = default_context if default_context is not None else {}
        if self.run_id:
            perf_logger.info(f"PerformanceLogger initialized for Run ID: {self.run_id}")
        else:
            perf_logger.info("PerformanceLogger initialized.")

    def start_timer(self, task_name: str):
        """
        Starts a timer for a given task.

        Args:
            task_name (str): A descriptive name for the task being timed.
        """
        self._start_times[task_name] = time.perf_counter()
        log_message = f"Task '{task_name}' started."
        self._log_with_context(log_message)

    def end_timer(self,
                  task_name: str,
                  records_processed: Optional[int] = None,
                  **extra_metrics: Any) -> Optional[float]:
        """
        Stops the timer for a given task, calculates the duration, and logs it.

        Args:
            task_name (str): The name of the task for which the timer was started.
            records_processed (Optional[int]): Number of records processed by the task.
            **extra_metrics: Additional key-value pairs to include in the log message.

        Returns:
            Optional[float]: The duration of the task in seconds, or None if the timer wasn't started.
        """
        start_time = self._start_times.pop(task_name, None)
        if start_time is None:
            perf_logger.warning(f"Timer for task '{task_name}' was not started or already ended.")
            return None

        duration = time.perf_counter() - start_time
        
        log_message = f"Task '{task_name}' ended. Duration: {duration:.4f} seconds."
        
        metrics_to_log = {}
        if records_processed is not None:
            metrics_to_log['records_processed'] = records_processed
            if duration > 0:
                metrics_to_log['records_per_second'] = f"{records_processed / duration:.2f}"
        
        if extra_metrics:
            metrics_to_log.update(extra_metrics)
        
        if metrics_to_log:
            log_message += " Metrics: " + ", ".join(f"{k}={v}" for k, v in metrics_to_log.items())
            
        self._log_with_context(log_message)
        return duration

    def _log_with_context(self, message: str, level: int = logging.INFO):
        """Helper to add context to log messages."""
        context_parts = []
        if self.run_id:
            context_parts.append(f"RunID='{self.run_id}'")
        
        # Add default context items
        for key, value in self.default_context.items():
            context_parts.append(f"{key}='{value}'")

        if context_parts:
            full_message = f"[{' | '.join(context_parts)}] {message}"
        else:
            full_message = message
        
        perf_logger.log(level, full_message)

    def log_metric(self, metric_name: str, value: Any, **context_tags: Any):
        """
        Logs a single arbitrary metric.

        Args:
            metric_name (str): Name of the metric.
            value (Any): Value of the metric.
            **context_tags: Additional key-value pairs for context.
        """
        log_message = f"Metric '{metric_name}': {value}."
        
        tags_to_log = {}
        if context_tags:
            tags_to_log.update(context_tags)
        
        if tags_to_log:
            log_message += " Tags: " + ", ".join(f"{k}={v}" for k, v in tags_to_log.items())
            
        self._log_with_context(log_message)


def time_it(perf_logger_instance: PerformanceLogger, task_name: Optional[str] = None, records_processed_func: Optional[callable] = None):
    """
    A decorator to easily time functions and log their performance.

    Args:
        perf_logger_instance (PerformanceLogger): An instance of PerformanceLogger.
        task_name (Optional[str]): Name of the task. If None, uses the function name.
        records_processed_func (Optional[callable]): A function that takes the
                                                     result of the decorated function
                                                     and returns the number of records processed.
                                                     Example: `lambda result_df: len(result_df)`
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            name_to_log = task_name if task_name else func.__name__
            perf_logger_instance.start_timer(name_to_log)
            try:
                result = func(*args, **kwargs)
                num_records = None
                if records_processed_func:
                    try:
                        num_records = records_processed_func(result)
                    except Exception as e:
                        perf_logger.warning(f"Error calling records_processed_func for task '{name_to_log}': {e}")
                perf_logger_instance.end_timer(name_to_log, records_processed=num_records)
                return result
            except Exception as e:
                # Log the error and re-raise to not alter program flow
                perf_logger.error(f"Exception in timed task '{name_to_log}': {e}", exc_info=True)
                # Ensure timer is ended even if an exception occurs, though duration might be misleading
                perf_logger_instance.end_timer(name_to_log, extra_metrics={"status": "failed", "error": str(e)})
                raise
        return wrapper
    return decorator


if __name__ == '__main__':
    # Configure basic logging for testing this module directly.
    # This setup sends 'performance_logger' logs to the console for easy viewing during tests.
    # In your actual application, 'performance_logger' would be configured by your main logging_config.yaml.
    
    # Setup a console handler for the performance_logger for this test
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Add handler to the specific performance logger
    # (and prevent propagation if you don't want root logger to also handle it)
    test_perf_logger_instance = logging.getLogger('performance_logger')
    test_perf_logger_instance.addHandler(console_handler)
    test_perf_logger_instance.setLevel(logging.INFO)
    test_perf_logger_instance.propagate = False # Optional: prevent double logging if root also has console handler

    # Also configure root logger for other general logs from this test script
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


    logger.info("--- Testing PerformanceLogger ---")

    # Initialize with a run ID
    p_log = PerformanceLogger(run_id="test_run_123", default_context={"environment": "dev"})

    # Simulate some tasks
    p_log.start_timer("DataExtraction")
    logger.info("Performing data extraction task...")
    time.sleep(0.5) # Simulate work
    p_log.end_timer("DataExtraction", records_processed=1000)

    p_log.start_timer("DataTransformation")
    logger.info("Performing data transformation task...")
    time.sleep(0.3) # Simulate work
    p_log.end_timer("DataTransformation", records_processed=950, quality_score=0.98)

    p_log.start_timer("DataLoading")
    logger.info("Performing data loading task...")
    time.sleep(0.7) # Simulate work
    p_log.end_timer("DataLoading", records_processed=950)

    # Test ending a timer that wasn't started
    logger.info("\n--- Testing non-started timer ---")
    p_log.end_timer("NonExistentTask")

    # Test logging a single metric
    logger.info("\n--- Testing single metric logging ---")
    p_log.log_metric("TotalMemoryUsageMB", 256.5, stage="PostProcessing")
    p_log.log_metric("APICallSuccessRate", 0.995)


    logger.info("\n--- Testing @time_it decorator ---")
    
    # Example usage of the decorator
    @time_it(p_log, task_name="DecoratedTaskA")
    def sample_task_a(duration):
        logger.info(f"Running {sample_task_a.__name__} for {duration}s...")
        time.sleep(duration)
        return "Task A complete"

    @time_it(p_log, records_processed_func=lambda res_list: len(res_list))
    def sample_task_b_with_records(num_items):
        logger.info(f"Running {sample_task_b_with_records.__name__} to generate {num_items} items...")
        time.sleep(0.2)
        return [i for i in range(num_items)]
    
    @time_it(p_log)
    def sample_task_c_with_error():
        logger.info(f"Running {sample_task_c_with_error.__name__}, which will raise an error.")
        time.sleep(0.1)
        raise ValueError("Simulated error in task C")

    sample_task_a(0.25)
    result_b = sample_task_b_with_records(5000)
    logger.info(f"Task B returned {len(result_b)} items.")

    try:
        sample_task_c_with_error()
    except ValueError as e:
        logger.info(f"Caught expected error from task C: {e}")


    logger.info("\nPerformanceLogger tests finished.")
    # Clean up handler for this test
    test_perf_logger_instance.removeHandler(console_handler)

