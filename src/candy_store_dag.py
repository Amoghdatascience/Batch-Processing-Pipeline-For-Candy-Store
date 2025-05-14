from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from pyspark.sql import SparkSession
from data_processor import DataProcessor
import os
from dotenv import load_dotenv
import shutil


def load_configuration(**kwargs):
    load_dotenv()
    config = {
        "mysql_url": os.getenv("MYSQL_URL"),
        "mysql_user": os.getenv("MYSQL_USER"),
        "mysql_password": os.getenv("MYSQL_PASSWORD"),
        "mongodb_uri": os.getenv("MONGODB_URI"),
        "mongodb_db": os.getenv("MONGO_DB"),
        "customers_table": os.getenv("CUSTOMERS_TABLE"),
        "products_table": os.getenv("PRODUCTS_TABLE"),
        "output_path": os.getenv("OUTPUT_PATH"),
        "mysql_connector_path": os.getenv("MYSQL_CONNECTOR_PATH"),
        "reload_inventory_daily": os.getenv("RELOAD_INVENTORY_DAILY", "false").lower()
        == "true",
        "start_date": os.getenv("MONGO_START_DATE", "20240201"),
        "end_date": os.getenv("MONGO_END_DATE", "20240210"),
    }
    ti = kwargs["ti"]
    ti.xcom_push(key="config", value=config)
    print("\n✅ Configuration Loaded & Pushed to XCom")


def create_spark_session(config):
    if not config.get("mysql_connector_path"):
        raise ValueError("❌ Missing MYSQL_CONNECTOR_PATH in configuration.")

    spark = (
        SparkSession.builder.appName("CandyStorePipeline")
        .config("spark.jars", config["mysql_connector_path"])
        .config(
            "spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1"
        )
        .config("spark.mongodb.input.uri", config["mongodb_uri"])
        .config("spark.mongodb.output.uri", config["mongodb_uri"])
        .getOrCreate()
    )
    return spark


def load_initial_data(**kwargs):
    ti = kwargs["ti"]
    config = ti.xcom_pull(task_ids="load_configuration", key="config")
    spark = create_spark_session(config)

    customers_df = (
        spark.read.format("jdbc")
        .options(
            url=config["mysql_url"],
            dbtable=config["customers_table"],
            user=config["mysql_user"],
            password=config["mysql_password"],
            driver="com.mysql.cj.jdbc.Driver",
        )
        .load()
    )

    products_df = (
        spark.read.format("jdbc")
        .options(
            url=config["mysql_url"],
            dbtable=config["products_table"],
            user=config["mysql_user"],
            password=config["mysql_password"],
            driver="com.mysql.cj.jdbc.Driver",
        )
        .load()
    )

    products_path = "/tmp/products.parquet"
    products_df.write.mode("overwrite").parquet(products_path)

    ti.xcom_push(key="products_table", value=products_path)
    print(f"\n✅ Saved products to {products_path} & pushed to XCom")


def process_batches(**kwargs):
    ti = kwargs["ti"]
    config = ti.xcom_pull(task_ids="load_configuration", key="config")
    products_path = ti.xcom_pull(task_ids="load_initial_data", key="products_table")

    print(f"\n🔎 Retrieved products_path from XCom: {products_path}")
    if not products_path:
        raise ValueError(
            "❌ products_table path not found in XCom from load_initial_data task"
        )

    spark = create_spark_session(config)
    data_processor = DataProcessor(spark)
    data_processor.configure(config)

    products_df = spark.read.parquet(products_path)
    data_processor.load_initial_inventory(products_df)

    start_date = datetime.strptime(config["start_date"], "%Y%m%d")
    end_date = datetime.strptime(config["end_date"], "%Y%m%d")
    batch_size = 10
    current_date = start_date

    all_orders = []
    all_order_line_items = []
    all_daily_summaries = []

    while current_date <= end_date:
        batch_dates = [
            (current_date + timedelta(days=i)).strftime("%Y%m%d")
            for i in range(batch_size)
            if (current_date + timedelta(days=i)) <= end_date
        ]

        print(f"\n📅 Processing batch for dates: {batch_dates}")

        for date_str in batch_dates:
            transactions_df = (
                spark.read.format("mongo")
                .option("database", config["mongodb_db"])
                .option("collection", f"transactions_{date_str}")
                .load()
            )

            if transactions_df.count() == 0:
                print(f"⚠️ No transactions for {date_str}, skipping.")
                continue

            if config["reload_inventory_daily"]:
                data_processor.load_initial_inventory(products_df)

            orders_df, order_line_items_df, cancelled_count = (
                data_processor.process_daily_transactions(transactions_df)
            )
            daily_summary = data_processor.calculate_daily_summary(
                order_line_items_df, date_str
            )

            all_orders.append(orders_df)
            all_order_line_items.append(order_line_items_df)
            all_daily_summaries.append(daily_summary)

        current_date += timedelta(days=batch_size)

    combined_orders = all_orders[0]
    for df in all_orders[1:]:
        combined_orders = combined_orders.union(df)

    combined_order_line_items = all_order_line_items[0]
    for df in all_order_line_items[1:]:
        combined_order_line_items = combined_order_line_items.union(df)

    combined_daily_summary = all_daily_summaries[0]
    for df in all_daily_summaries[1:]:
        combined_daily_summary = combined_daily_summary.union(df)

    combined_orders.write.mode("overwrite").parquet("/tmp/combined_orders.parquet")
    combined_order_line_items.write.mode("overwrite").parquet(
        "/tmp/combined_order_line_items.parquet"
    )
    combined_daily_summary.write.mode("overwrite").parquet(
        "/tmp/combined_daily_summary.parquet"
    )

    print("\n✅ Batch processing complete")


def safe_to_csv(spark_df, file_path):
    """
    Safely write Spark DataFrame to CSV using Pandas after cleaning up conflicting directories.
    If a folder with the same name exists (due to Spark's write.csv()), it is removed.
    """
    if os.path.isdir(file_path):
        shutil.rmtree(file_path)

    spark_df.toPandas().to_csv(file_path, index=False)
    print(f"✅ Saved: {file_path}")


def save_outputs(**kwargs):
    ti = kwargs["ti"]

    config = ti.xcom_pull(task_ids="load_configuration", key="config")
    products_path = ti.xcom_pull(task_ids="load_initial_data", key="products_table")
    output_path = config["output_path"]

    if not products_path:
        raise ValueError(
            "products_table path not found in XCom from load_initial_data task"
        )

    spark = (
        SparkSession.builder.appName("CandyStoreSaveOutputs")
        .config("spark.jars", config["mysql_connector_path"])
        .getOrCreate()
    )

    data_processor = DataProcessor(spark)
    data_processor.configure(config)

    products_df = spark.read.parquet(products_path)
    data_processor.load_initial_inventory(products_df)

    combined_orders = spark.read.parquet("/tmp/combined_orders.parquet")
    combined_order_line_items = spark.read.parquet(
        "/tmp/combined_order_line_items.parquet"
    )
    combined_daily_summary = spark.read.parquet("/tmp/combined_daily_summary.parquet")

    os.makedirs(output_path, exist_ok=True)

    safe_to_csv(combined_orders, os.path.join(output_path, "orders.csv"))
    safe_to_csv(
        combined_order_line_items, os.path.join(output_path, "order_line_items.csv")
    )
    safe_to_csv(combined_daily_summary, os.path.join(output_path, "daily_summary.csv"))

    final_inventory = data_processor.current_inventory.select(
        "product_id", "product_name", "current_stock"
    )
    safe_to_csv(final_inventory, os.path.join(output_path, "products_updated.csv"))

    print("\n✅ All outputs saved successfully to:", output_path)


default_args = {
    "owner": "airflow",
    "start_date": datetime(2025, 3, 4),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "candy_store_pipeline",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
) as dag:

    load_configuration_task = PythonOperator(
        task_id="load_configuration", python_callable=load_configuration
    )
    load_initial_data_task = PythonOperator(
        task_id="load_initial_data",
        python_callable=load_initial_data,
        provide_context=True,
    )
    process_batches_task = PythonOperator(
        task_id="process_batches", python_callable=process_batches, provide_context=True
    )
    save_outputs_task = PythonOperator(
        task_id="save_outputs", python_callable=save_outputs, provide_context=True
    )

    (
        load_configuration_task
        >> load_initial_data_task
        >> process_batches_task
        >> save_outputs_task
    )
