from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    explode,
    col,
    round as spark_round,
    sum as spark_sum,
    count,
    abs as spark_abs,
)
from typing import Dict, Tuple
import os
import glob
import shutil
import decimal
import numpy as np
from time_series import ProphetForecaster
from datetime import datetime, timedelta
from pyspark.sql.types import DoubleType, DecimalType
from pyspark.sql.functions import col, sum as spark_sum, count, lit, when
from pyspark.sql.functions import (
    col,
    countDistinct,
    date_format,
    to_date,
    round as spark_round,
    concat,
    substr,
)
import csv


class DataProcessor:
    def __init__(self, spark: SparkSession):
        self.spark = spark
        # Initialize all class properties
        self.config = None
        self.current_inventory = None
        self.inventory_initialized = False
        self.original_products_df = None  # Store original products data
        self.reload_inventory_daily = False  # New flag for inventory reload
        self.order_items = None
        self.products_df = None
        self.customers_df = None
        self.transactions_df = None
        self.orders_df = None
        self.order_line_items_df = None
        self.daily_summary_df = None
        self.total_cancelled_items = 0
        self_inventory_dict = {}

    def configure(self, config: Dict) -> None:
        """Configure the data processor with environment settings"""
        self.config = config
        self.reload_inventory_daily = config.get("reload_inventory_daily", False)
        print("\nINITIALIZING DATA SOURCES")
        print("-" * 80)
        if self.reload_inventory_daily:
            print("Daily inventory reload: ENABLED")
        else:
            print("Daily inventory reload: DISABLED")

    def finalize_processing(self) -> None:
        """Finalize processing and create summary"""
        print("\nPROCESSING COMPLETE")
        print("=" * 80)
        # print(f"Total Cancelled Items: {self.total_cancelled_items}")

    # ------------------------------------------------------------------------------------------------
    # Try not to change the logic of the time series forecasting model
    # DO NOT change functions with prefix _
    # ------------------------------------------------------------------------------------------------
    def forecast_sales_and_profits(
        self, daily_summary_df: DataFrame, forecast_days: int = 1
    ) -> DataFrame:
        """
        Main forecasting function that coordinates the forecasting process
        """
        try:
            # Build model
            model_data = self.build_time_series_model(daily_summary_df)

            # Calculate accuracy metrics
            metrics = self.calculate_forecast_metrics(model_data)

            # Generate forecasts
            forecast_df = self.make_forecasts(model_data, forecast_days)

            return forecast_df

        except Exception as e:
            print(
                f"Error in forecast_sales_and_profits: {str(e)}, please check the data"
            )
            return None

    def print_inventory_levels(self) -> None:
        """Print current inventory levels for all products"""
        print("\nCURRENT INVENTORY LEVELS")
        print("-" * 40)

        inventory_data = self.current_inventory.orderBy("product_id").collect()
        for row in inventory_data:
            print(
                f"• {row['product_name']:<30} (ID: {row['product_id']:>3}): {row['current_stock']:>4} units"
            )
        print("-" * 40)

    def build_time_series_model(self, daily_summary_df: DataFrame) -> dict:
        """Build Prophet models for sales and profits"""
        print("\n" + "=" * 80)
        print("TIME SERIES MODEL CONSTRUCTION")
        print("-" * 80)

        model_data = self._prepare_time_series_data(daily_summary_df)
        return self._fit_forecasting_models(model_data)

    def calculate_forecast_metrics(self, model_data: dict) -> dict:
        """Calculate forecast accuracy metrics for both models"""
        print("\nCalculating forecast accuracy metrics...")

        # Get metrics from each model
        sales_metrics = model_data["sales_model"].get_metrics()
        profit_metrics = model_data["profit_model"].get_metrics()

        metrics = {
            "sales_mae": sales_metrics["mae"],
            "sales_mse": sales_metrics["mse"],
            "profit_mae": profit_metrics["mae"],
            "profit_mse": profit_metrics["mse"],
        }

        # Print metrics and model types
        print("\nForecast Error Metrics:")
        print(f"Sales Model Type: {sales_metrics['model_type']}")
        print(f"Sales MAE: ${metrics['sales_mae']:.2f}")
        print(f"Sales MSE: ${metrics['sales_mse']:.2f}")
        print(f"Profit Model Type: {profit_metrics['model_type']}")
        print(f"Profit MAE: ${metrics['profit_mae']:.2f}")
        print(f"Profit MSE: ${metrics['profit_mse']:.2f}")

        return metrics

    def make_forecasts(self, model_data: dict, forecast_days: int = 7) -> DataFrame:
        """Generate forecasts using Prophet models"""
        print(f"\nGenerating {forecast_days}-day forecast...")

        forecasts = self._generate_model_forecasts(model_data, forecast_days)
        forecast_dates = self._generate_forecast_dates(
            model_data["training_data"]["dates"][-1], forecast_days
        )

        return self._create_forecast_dataframe(forecast_dates, forecasts)

    def _prepare_time_series_data(self, daily_summary_df: DataFrame) -> dict:
        """Prepare data for time series modeling"""
        data = (
            daily_summary_df.select("date", "total_sales", "total_profit")
            .orderBy("date")
            .collect()
        )

        dates = np.array([row["date"] for row in data])
        sales_series = np.array([float(row["total_sales"]) for row in data])
        profit_series = np.array([float(row["total_profit"]) for row in data])

        self._print_dataset_info(dates, sales_series, profit_series)

        return {"dates": dates, "sales": sales_series, "profits": profit_series}

    def _print_dataset_info(
        self, dates: np.ndarray, sales: np.ndarray, profits: np.ndarray
    ) -> None:
        """Print time series dataset information"""
        print("Dataset Information:")
        print(f"• Time Period:          {dates[0]} to {dates[-1]}")
        print(f"• Number of Data Points: {len(dates)}")
        print(f"• Average Daily Sales:   ${np.mean(sales):.2f}")
        print(f"• Average Daily Profit:  ${np.mean(profits):.2f}")

    def _fit_forecasting_models(self, data: dict) -> dict:
        """Fit Prophet models to the prepared data"""
        print("\nFitting Models...")
        sales_forecaster = ProphetForecaster()
        profit_forecaster = ProphetForecaster()

        sales_forecaster.fit(data["sales"])
        profit_forecaster.fit(data["profits"])
        print("Model fitting completed successfully")
        print("=" * 80)

        return {
            "sales_model": sales_forecaster,
            "profit_model": profit_forecaster,
            "training_data": data,
        }

    def _generate_model_forecasts(self, model_data: dict, forecast_days: int) -> dict:
        """Generate forecasts from both models"""
        return {
            "sales": model_data["sales_model"].predict(forecast_days),
            "profits": model_data["profit_model"].predict(forecast_days),
        }

    def _generate_forecast_dates(self, last_date: datetime, forecast_days: int) -> list:
        """Generate dates for the forecast period"""
        return [last_date + timedelta(days=i + 1) for i in range(forecast_days)]

    def _create_forecast_dataframe(self, dates: list, forecasts: dict) -> DataFrame:
        """Create Spark DataFrame from forecast data"""
        forecast_rows = [
            (date, float(sales), float(profits))
            for date, sales, profits in zip(
                dates, forecasts["sales"], forecasts["profits"]
            )
        ]

        return self.spark.createDataFrame(
            forecast_rows, ["date", "forecasted_sales", "forecasted_profit"]
        )

    def load_initial_inventory(self, products_df: DataFrame):
        """
        Load initial inventory into a DataFrame and a Python dictionary for fast lookups.
        """
        self.current_inventory = products_df.select(
            "product_id",
            "product_name",
            col("stock").alias("current_stock"),
            col("sales_price").alias("unit_price"),
            col("cost_to_make").alias("unit_cost"),
        )

        self.inventory_dict = {
            row["product_id"]: {
                "product_name": row["product_name"],
                "current_stock": row["current_stock"],
                "unit_price": row["unit_price"],
                "unit_cost": row["unit_cost"],
            }
            for row in self.current_inventory.collect()
        }

    def process_order_line_item(
        self, order_id: int, product_id: int, product_name: str, requested_qty: int
    ) -> Tuple[int, float, bool]:
        """
        Process one order line item (a product in an order).
        Check inventory via the fast Python dictionary, not via Spark.
        """
        product = self.inventory_dict.get(product_id)

        if not product:
            print(f"WARNING: Product ID {product_id} not found. Cancelling item.")
            return 0, 0.0, True  # Cancelled item

        available_stock = product["current_stock"]
        unit_price = product["unit_price"]

        if available_stock >= requested_qty:
            # Fully fulfill the item
            approved_qty = requested_qty
            line_total = approved_qty * unit_price

            self.inventory_dict[product_id]["current_stock"] -= approved_qty
            return approved_qty, line_total, False  # Not cancelled
        else:

            print(
                f"INSUFFICIENT STOCK: {product_name} (ID {product_id}) - Requested: {requested_qty}, Available: {available_stock}. Cancelling."
            )
            return 0, 0.0, True

    #   def _update_inventory(self, product_id: int, new_stock: int):
    #       self.current_inventory = self.current_inventory.withColumn(
    #           "current_stock",
    #          when(col("product_id") == product_id, safe_stock).otherwise(col("current_stock"))
    #      )

    def process_daily_transactions(
        self, transactions_df: DataFrame
    ) -> Tuple[DataFrame, DataFrame, int]:

        orders = []
        order_line_items = []
        cancelled_items_count = 0

        for row in transactions_df.collect():
            order_id = row["transaction_id"]
            customer_id = row["customer_id"]
            timestamp = row["timestamp"]
            product_id = row["product_id"]
            product_name = row["product_name"]
            requested_qty = row["qty"]

            if requested_qty is None:
                continue

            approved_qty, line_total, is_cancelled = self.process_order_line_item(
                order_id, product_id, product_name, requested_qty
            )

            order_line_items.append(
                (order_id, product_id, product_name, approved_qty, line_total)
            )

            if is_cancelled:
                cancelled_items_count += 1

            if order_id not in [o[0] for o in orders]:
                orders.append((order_id, customer_id, timestamp))

        orders_df = self.spark.createDataFrame(
            orders, ["order_id", "customer_id", "timestamp"]
        )
        order_line_items_df = self.spark.createDataFrame(
            order_line_items,
            ["order_id", "product_id", "product_name", "quantity", "line_total"],
        )

        return orders_df, order_line_items_df, cancelled_items_count

    def calculate_daily_summary(
        self, order_line_items_df: DataFrame, date: str
    ) -> DataFrame:
        """
        Calculate daily sales and profit for forecasting.
        """
        items_with_costs = order_line_items_df.join(
            self.current_inventory.select("product_id", "unit_price", "unit_cost"),
            on="product_id",
            how="left",
        )

        items_with_costs = items_with_costs.withColumn(
            "profit", (col("unit_price") - col("unit_cost")) * col("quantity")
        )

        daily_summary = items_with_costs.agg(
            spark_sum("line_total").alias("total_sales"),
            spark_sum("profit").alias("total_profit"),
        ).withColumn("date", lit(date))

        return daily_summary.select("date", "total_sales", "total_profit")

    def generate_orders_summary(
        self, transactions_df: DataFrame, order_line_items_df: DataFrame
    ) -> DataFrame:

        revenue_summary = order_line_items_df.groupBy("order_id").agg(
            spark_sum("line_total").alias("total_revenue")
        )

        total_items_summary = order_line_items_df.groupBy("order_id").agg(
            countDistinct("product_id").alias("num_items")
        )

        orders_summary = transactions_df.select(
            "transaction_id", "customer_id", "timestamp"
        ).distinct()

        orders_summary = (
            orders_summary.join(
                revenue_summary,
                orders_summary.transaction_id == revenue_summary.order_id,
                "left",
            )
            .join(
                total_items_summary,
                orders_summary.transaction_id == total_items_summary.order_id,
                "left",
            )
            .select(
                col("transaction_id").alias("order_id"),
                "customer_id",
                "timestamp",
                "total_revenue",
                "num_items",
            )
            .fillna({"total_revenue": 0.0, "num_items": 0})
        )

        orders_summary = orders_summary.withColumnRenamed(
            "timestamp", "order_datetime"
        ).withColumnRenamed("total_revenue", "total_amount")

        orders_summary = orders_summary.select(
            "order_id", "order_datetime", "customer_id", "total_amount", "num_items"
        )

        return orders_summary

    def refresh_inventory_from_dict(self):
        """
        Recreate current_inventory DataFrame from the in-memory dictionary.
        This can be used if the DataFrame is needed for reporting after daily processing.
        """
        inventory_rows = [
            (
                pid,
                pdata["product_name"],
                pdata["current_stock"],
                pdata["unit_price"],
                pdata["unit_cost"],
            )
            for pid, pdata in self.inventory_dict.items()
        ]
        self.current_inventory = self.spark.createDataFrame(
            inventory_rows,
            ["product_id", "product_name", "current_stock", "unit_price", "unit_cost"],
        )

    def save_to_csv(self, df: DataFrame, output_path: str, filename: str):
        """
        Save any DataFrame to CSV.
        """
        file_path = os.path.join(output_path, filename)
        df.coalesce(1).write.csv(file_path, header=True, mode="overwrite")

    def finalize_processing(self):
        """
        Final processing summary.
        """
        print("\nPROCESSING COMPLETE")
        print("=" * 80)

    def prepare_order_line_items_for_csv(
        self, order_line_items_df: DataFrame
    ) -> DataFrame:

        order_line_items_with_prices = order_line_items_df.join(
            self.current_inventory.select("product_id", "unit_price"),
            on="product_id",
            how="left",
        )

        return order_line_items_with_prices.select(
            "order_id", "product_id", "quantity", "unit_price", "line_total"
        ).orderBy("order_id", "product_id")

    def generate_daily_summary(
        self, all_orders_df: DataFrame, all_daily_summaries_df: DataFrame
    ) -> DataFrame:
        """
        Generate daily summary with correct date handling and order count.
        """

        all_daily_summaries_df = all_daily_summaries_df.withColumn(
            "date", date_format(to_date(col("date"), "yyyyMMdd"), "yyyy-MM-dd")
        )

        orders_with_date = all_orders_df.withColumn(
            "date", date_format(col("order_datetime"), "yyyy-MM-dd")
        )

        daily_order_counts = orders_with_date.groupBy("date").agg(
            countDistinct("order_id").alias("num_orders")
        )

        daily_summary = all_daily_summaries_df.join(
            daily_order_counts, on="date", how="left"
        ).fillna({"num_orders": 0})

        daily_summary = daily_summary.withColumn(
            "total_sales", spark_round(col("total_sales"), 2)
        ).withColumn("total_profit", spark_round(col("total_profit"), 2))

        # Final select and order
        daily_summary = daily_summary.select(
            "date", "num_orders", "total_sales", "total_profit"
        ).orderBy("date")

        return daily_summary

    def save_final_inventory(self, output_path: str):

        sorted_inventory = self.current_inventory.orderBy("product_id")

        products_updated = sorted_inventory.select(
            "product_id", "product_name", "current_stock"
        )

        self.save_to_csv(products_updated, output_path, "products_updated.csv")

        print(" Final inventory saved as products_updated.csv")

    def save_daily_summary_to_csv(daily_summary_df, output_path: str):

        daily_summary_file = os.path.join(output_path, "daily_summary.csv")

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
