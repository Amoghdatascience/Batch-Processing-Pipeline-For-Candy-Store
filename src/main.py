from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, explode
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    ArrayType,
)
from data_processor import DataProcessor
from dotenv import load_dotenv
from datetime import datetime, timedelta
import os
from typing import Dict, Tuple
import traceback
from data_processor import DataProcessor
from datetime import datetime, timedelta
from time_series import ProphetForecaster
import os
import csv
import shutil


def create_spark_session(app_name: str = "CandyStoreAnalytics") -> SparkSession:
    """
    Create and configure Spark session with MongoDB and MySQL connectors
    """
    load_dotenv()
    mysql_connector_path = os.getenv("MYSQL_CONNECTOR_PATH")
    mongodb_uri = os.getenv("MONGODB_URI")
    mongodb_db = os.getenv("MONGO_DB")

    print(f" MYSQL_CONNECTOR_PATH = {mysql_connector_path}")
    print(f" MONGODB_URI = {mongodb_uri}")
    print(f" MONGODB_DB = {mongodb_db}")

    if not mysql_connector_path:
        raise ValueError(
            " ERROR: MYSQL_CONNECTOR_PATH is not set. Check your .env file or system environment variables."
        )
    if not mongodb_uri:
        raise ValueError(
            " ERROR: MONGODB_URI is not set. Check your .env file or system environment variables."
        )
    if not mongodb_db:
        raise ValueError(
            " ERROR: MONGO_DB is not set. Check your .env file or system environment variables."
        )

    return (
        SparkSession.builder.appName(app_name)
        .config(
            "spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1"
        )
        .config("spark.jars", mysql_connector_path)
        .config("spark.mongodb.input.uri", mongodb_uri)
        .config("spark.mongodb.output.uri", mongodb_uri)
        .getOrCreate()
    )


def setup_configuration() -> Tuple[Dict, list]:
    """Setup application configuration"""
    load_dotenv()
    config = {
        "mongodb_uri": os.getenv("MONGODB_URI"),
        "mongodb_db": os.getenv("MONGO_DB"),
        "mysql_url": os.getenv("MYSQL_URL"),
        "mysql_user": os.getenv("MYSQL_USER"),
        "mysql_password": os.getenv("MYSQL_PASSWORD"),
        "mysql_db": os.getenv("MYSQL_DB"),
        "customers_table": os.getenv("CUSTOMERS_TABLE"),
        "products_table": os.getenv("PRODUCTS_TABLE"),
        "output_path": os.getenv("OUTPUT_PATH"),
        "reload_inventory_daily": os.getenv("RELOAD_INVENTORY_DAILY", "false").lower()
        == "true",
    }

    start_date = os.getenv("MONGO_START_DATE", "20240201")
    end_date = os.getenv("MONGO_END_DATE", "20240210")
    date_range = get_date_range(start_date, end_date)

    return config, date_range


def get_date_range(start_date: str, end_date: str) -> list:
    """Generates a list of dates between start and end date"""
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    return [
        (start + timedelta(days=i)).strftime("%Y%m%d")
        for i in range((end - start).days + 1)
    ]


def load_csv_to_mysql(
    spark: SparkSession, file_path: str, table_name: str, config: Dict
):
    """Load a CSV file into MySQL using PySpark"""
    if not os.path.exists(file_path):
        print(f" ERROR: File {file_path} does not exist. Skipping {table_name} table.")
        return

    print(f"\nLOADING CSV INTO MYSQL TABLE: {table_name}")
    print("-" * 80)
    try:
        df = spark.read.csv(file_path, header=True, inferSchema=True)

        df.write.format("jdbc").option("url", config["mysql_url"]).option(
            "dbtable", table_name
        ).option("user", config["mysql_user"]).option(
            "password", config["mysql_password"]
        ).option(
            "driver", "com.mysql.cj.jdbc.Driver"
        ).mode(
            "overwrite"
        ).save()

        print(f" Successfully loaded {file_path} into {table_name}.")
    except Exception as e:
        print(f" Error loading {file_path} into MySQL: {str(e)}")
        print(traceback.format_exc())


def load_json_to_mongodb(
    spark: SparkSession, file_path: str, collection_name: str, config: Dict
):
    """Loading a JSON file into MongoDB using PySpark"""
    if not os.path.exists(file_path):
        print(
            f" ERROR: File {file_path} does not exist. Skipping {collection_name} collection."
        )
        return

    print(f"\nLOADING JSON INTO MONGODB COLLECTION: {collection_name}")
    print("-" * 80)

    try:
        df = spark.read.option("multiline", "true").json(file_path)
        df = df.withColumn("items", explode(col("items")))
        df = df.select(
            col("transaction_id"),
            col("customer_id"),
            col("timestamp"),
            col("items.product_id"),
            col("items.product_name"),
            col("items.qty"),
        ).filter(col("qty").isNotNull())

        df.write.format("mongo").option("uri", config["mongodb_uri"]).option(
            "database", config["mongodb_db"]
        ).option("collection", collection_name).mode("overwrite").save()

        print(f" Successfully loaded {file_path} into {collection_name}.")
    except Exception as e:
        print(f" Error loading {file_path} into MongoDB: {str(e)}")
        print(traceback.format_exc())


def load_mysql_to_spark(spark: SparkSession, config: Dict, table_name: str):
    """Load MySQL table into Spark DataFrame and display preview"""
    print(f"\nLOADING MYSQL TABLE INTO SPARK SESSION: {table_name}")
    print("-" * 80)
    df = (
        spark.read.format("jdbc")
        .option("url", config["mysql_url"])
        .option("dbtable", table_name)
        .option("user", config["mysql_user"])
        .option("password", config["mysql_password"])
        .option("driver", "com.mysql.cj.jdbc.Driver")
        .load()
    )

    print(f" MySQL Table {table_name} Row Count: {df.count()}")


def load_mongodb_to_spark(
    spark: SparkSession, collection_name: str, config: Dict
) -> DataFrame:
    """
    Load data from a MongoDB collection into a Spark DataFrame
    """
    print(f"\nLOADING DATA FROM MONGODB COLLECTION: {collection_name}")
    print("-" * 80)

    try:
        df = (
            spark.read.format("mongo")
            .option("uri", config["mongodb_uri"])
            .option("database", config["mongodb_db"])
            .option("collection", collection_name)
            .load()
        )

        # df.show(5)
        # print(f"DataFrame Dimensions: ({df.count()}, {len(df.columns)})")

        return df
    except Exception as e:
        print(
            f" Error loading data from MongoDB collection {collection_name}: {str(e)}"
        )
        print(traceback.format_exc())
        return None


def initialize_data_processor(spark: SparkSession, config: Dict) -> DataProcessor:
    """Initialize and configure the DataProcessor"""
    print("\nINITIALIZING DATA SOURCES")
    print("-" * 80)

    data_processor = DataProcessor(spark)
    data_processor.config = config
    return data_processor


def print_processing_complete(total_cancelled_items: int) -> None:
    """Print processing completion message"""
    print("\nPROCESSING COMPLETE")
    print("=" * 80)
    print(f"Total Cancelled Items: {total_cancelled_items}")


def print_daily_summary(orders_df, order_items_df, cancelled_count):
    """Print summary of daily processing"""
    processed_items = order_items_df.filter(col("quantity") > 0).count()
    print("\nDAILY PROCESSING SUMMARY")
    print("-" * 40)
    print(f"• Successfully Processed Orders: {orders_df.count()}")
    print(f"• Successfully Processed Items: {processed_items}")
    print(f"• Items Cancelled (Inventory): {cancelled_count}")
    print("-" * 40)


def generate_forecasts(
    data_processor: DataProcessor, final_daily_summary, output_path: str
):
    """Generate and save sales forecasts"""
    print("\nGENERATING FORECASTS")
    print("-" * 80)

    try:
        if final_daily_summary is not None and final_daily_summary.count() > 0:
            print("Schema before forecasting:", final_daily_summary.printSchema())
            forecast_df = data_processor.forecast_sales_and_profits(final_daily_summary)
            if forecast_df is not None:
                data_processor.save_to_csv(
                    forecast_df, output_path, "sales_profit_forecast.csv"
                )
        else:
            print("Warning: No daily summary data available for forecasting")
    except Exception as e:
        print(f"Warning: Could not generate forecasts: {str(e)}")
        print("Stack trace:", traceback.format_exc())


def forecast_sales_and_profits_from_df(daily_summary_df, output_path: str):

    summary_data = daily_summary_df.select(
        "date", "total_sales", "total_profit"
    ).collect()

    dates = []
    sales_data = []
    profit_data = []

    for row in summary_data:
        dates.append(row["date"])
        sales_data.append(float(row["total_sales"]))
        profit_data.append(float(row["total_profit"]))

    last_date = datetime.strptime(dates[-1], "%Y-%m-%d")

    forecast_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")

    sales_forecaster = ProphetForecaster()
    profit_forecaster = ProphetForecaster()

    sales_forecaster.fit(sales_data)
    profit_forecaster.fit(profit_data)

    sales_forecaster.get_metrics()
    profit_forecaster.get_metrics()

    forecasted_sales = sales_forecaster.predict(1)[0]
    forecasted_profit = profit_forecaster.predict(1)[0]

    forecast_file = os.path.join(output_path, "sales_profit_forecast.csv")
    with open(forecast_file, mode="w", newline="") as file:
        file.write("date,forecasted_sales,forecasted_profit\n")
        file.write(
            f"{forecast_date},{round(forecasted_sales, 2)},{round(forecasted_profit, 2)}\n"
        )

    print(f"\n 1-Day Sales & Profit Forecast saved to {forecast_file}\n")


def save_daily_summary_to_csv(daily_summary_df, output_path: str):
    """
    Save daily_summary DataFrame to daily_summary.csv using Python's csv module.
    """
    daily_summary_file = os.path.join(output_path, "daily_summary.csv")

    if os.path.isdir(daily_summary_file):
        shutil.rmtree(daily_summary_file)

    rows = daily_summary_df.collect()

    with open(daily_summary_file, mode="w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(["date", "num_orders", "total_sales", "total_profit"])

        for row in rows:
            writer.writerow(
                [
                    row["date"],
                    row["num_orders"],
                    round(row["total_sales"], 2),
                    round(row["total_profit"], 2),
                ]
            )

    print(f"\n Saved daily_summary.csv to {daily_summary_file}")


def save_products_updated_to_csv(products_updated_df, output_path: str):
    """
    Save products_updated DataFrame to products_updated.csv using Python csv module.
    """
    products_updated_file = os.path.join(output_path, "products_updated.csv")

    if os.path.isdir(products_updated_file):
        shutil.rmtree(products_updated_file)

    rows = products_updated_df.collect()

    with open(products_updated_file, mode="w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(["product_id", "product_name", "current_stock"])

        for row in rows:
            writer.writerow(
                [row["product_id"], row["product_name"], row["current_stock"]]
            )

    print(f"\n Saved products_updated.csv to {products_updated_file}")


def save_order_line_items_to_csv(prepared_order_line_items_df, output_path: str):
    """
    Save prepared_order_line_items DataFrame to order_line_items.csv using Python's csv module.
    """
    order_line_items_file = os.path.join(output_path, "order_line_items.csv")

    if os.path.isdir(order_line_items_file):
        shutil.rmtree(order_line_items_file)

    rows = prepared_order_line_items_df.collect()

    with open(order_line_items_file, mode="w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(
            ["order_id", "product_id", "quantity", "unit_price", "line_total"]
        )

        for row in rows:
            writer.writerow(
                [
                    row["order_id"],
                    row["product_id"],
                    row["quantity"],
                    row["unit_price"],
                    round(row["line_total"], 2),
                ]
            )

    print(f"\n Saved prepared order line items to {order_line_items_file}")


def save_order_line_items_to_csv(prepared_order_line_items_df, output_path: str):

    order_line_items_file = os.path.join(output_path, "order_line_items.csv")

    if os.path.isdir(order_line_items_file):
        shutil.rmtree(order_line_items_file)

    rows = prepared_order_line_items_df.collect()

    with open(order_line_items_file, mode="w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(
            ["order_id", "product_id", "quantity", "unit_price", "line_total"]
        )

        for row in rows:
            writer.writerow(
                [
                    row["order_id"],
                    row["product_id"],
                    row["quantity"],
                    row["unit_price"],
                    round(row["line_total"], 2),
                ]
            )

    print(f"\n Saved prepared order line items to {order_line_items_file}")


def save_combined_orders_to_csv(combined_orders_df, output_path: str):

    orders_file = os.path.join(output_path, "orders.csv")

    if os.path.isdir(orders_file):
        shutil.rmtree(orders_file)

    rows = combined_orders_df.collect()

    with open(orders_file, mode="w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(
            ["order_id", "order_datetime", "customer_id", "total_amount", "num_items"]
        )

        for row in rows:
            writer.writerow(
                [
                    row["order_id"],
                    row["order_datetime"],
                    row["customer_id"],
                    round(row["total_amount"], 2),
                    row["num_items"],
                ]
            )
    print(f"\nSaved combined orders to {orders_file}")


def main():
    config, date_range = setup_configuration()
    print("\nPROCESSING PERIOD")
    print("-" * 80)
    print(f"Start Date: {date_range[0]}")
    print(f"End Date:   {date_range[-1]}")
    print("=" * 80)
    spark = create_spark_session()
    data_processor = DataProcessor(spark)
    data_processor.configure(config)
    customers_df = (
        spark.read.format("jdbc")
        .option("url", config["mysql_url"])
        .option("dbtable", config["customers_table"])
        .option("user", config["mysql_user"])
        .option("password", config["mysql_password"])
        .option("driver", "com.mysql.cj.jdbc.Driver")
        .load()
    )
    products_df = (
        spark.read.format("jdbc")
        .option("url", config["mysql_url"])
        .option("dbtable", config["products_table"])
        .option("user", config["mysql_user"])
        .option("password", config["mysql_password"])
        .option("driver", "com.mysql.cj.jdbc.Driver")
        .load()
    )
    data_processor.load_initial_inventory(products_df)
    all_order_line_items = []
    all_orders = []
    daily_summaries = []
    for date in date_range:
        print(f"\nPROCESSING TRANSACTIONS FOR DATE: {date}")
        print("-" * 80)

        transactions_df = load_mongodb_to_spark(spark, f"transactions_{date}", config)
        if transactions_df is None or transactions_df.count() == 0:
            print(f"No transactions found for {date}. Skipping.")
            continue

        if config["reload_inventory_daily"]:
            data_processor.load_initial_inventory(products_df)

        orders_df, order_line_items_df, cancelled_count = (
            data_processor.process_daily_transactions(transactions_df)
        )

        print_daily_summary(orders_df, order_line_items_df, cancelled_count)

        all_order_line_items.append(order_line_items_df)
        orders_summary_df = data_processor.generate_orders_summary(
            transactions_df, order_line_items_df
        )
        all_orders.append(orders_summary_df)

        daily_summary = data_processor.calculate_daily_summary(
            order_line_items_df, date
        )
        daily_summaries.append(daily_summary)
    combined_order_line_items = all_order_line_items[0]
    for item_df in all_order_line_items[1:]:
        combined_order_line_items = combined_order_line_items.union(item_df)
    combined_orders = all_orders[0]
    for order_df in all_orders[1:]:
        combined_orders = combined_orders.union(order_df)
    prepared_order_line_items = data_processor.prepare_order_line_items_for_csv(
        combined_order_line_items
    )
    save_order_line_items_to_csv(prepared_order_line_items, config["output_path"])
    save_combined_orders_to_csv(
        combined_orders.orderBy("order_id"), config["output_path"]
    )
    final_daily_summary = daily_summaries[0]
    for summary in daily_summaries[1:]:
        final_daily_summary = final_daily_summary.union(summary)

    daily_summary_df = data_processor.generate_daily_summary(
        combined_orders, final_daily_summary
    )
    print("\nDaily summary saved to daily_summary.csv")
    data_processor.refresh_inventory_from_dict()
    data_processor.save_final_inventory(config["output_path"])
    daily_summary_df = data_processor.generate_daily_summary(
        combined_orders, final_daily_summary
    )
    save_daily_summary_to_csv(daily_summary_df, config["output_path"])
    data_processor.refresh_inventory_from_dict()
    save_products_updated_to_csv(
        data_processor.current_inventory.orderBy("product_id"), config["output_path"]
    )
    forecast_sales_and_profits_from_df(daily_summary_df, config["output_path"])
    data_processor.finalize_processing()
    print("\nCleaning up...")
    spark.stop()


if __name__ == "__main__":
    main()
