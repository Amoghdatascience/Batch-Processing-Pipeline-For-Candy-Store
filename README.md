## Implementation of Batch Processing Pipeline for Candy Store

This project is a batch processing pipeline for a fictional candy store from RIT that processes daily sales transactions, manages inventory, generates summary reports, and forecasts future sales and profits using machine learning (Prophet). The pipeline handles data ingestion, order processing, inventory updates, and forecasting using PySpark, MySQL, MongoDB, and Apache Airflow for orchestration.


### Changes Made

- Implemented batch processing pipeline using PySpark to process daily orders, update inventory, and generate reports.
- Integrated MySQL and MongoDB data sources for customers, products, and transactions.
- Created order processing logic to validate, fulfill, or cancel orders based on inventory.
- Developed logic to compute daily sales, profit, and order counts.
- Built time series forecasting model using Prophet to predict next-day sales and profit.
- Created Airflow DAG (candy_store_dag.py) to automate the entire batch process.
- Applied consistent date formatting using date_format and to_date.
- Removed duplicate call to forecast metrics printing.
- Added Python-native csv.writer functions to save orders.csv, order_line_items.csv, and products_updated.csv.
- Adjusted forecast_sales_and_profits_from_df to handle formatted dates correctly.
- Implemented CSV writers for:
1. daily_summary.csv
2. order_line_items.csv
3. orders.csv
4. products_updated.csv
5. sales_profit_forecast.csv

### Technologies Used:
- Python (3.12)
- PySpark
- Prophet
- MySQL
- MongoDB
- Apache Airflow
- GitLab CI/CD
- CSV Files

### Testing
- Ran Batch Processing manually using main.py and verified that:
1. All orders processed correctly.
2. Inventory updated accurately
3. All required files are generated.
4. Forecasting metrics printed correctly


### Checklist
- [x] Code formatted with black
- [x] All tests passing
- [x] No sensitive data included
- [x] Only modified necessary files


### Contact and Contributions
Feel free to fork this repository to contribute. Pull requests are welcome. This project was done as a part of Coursework under Prof. Zimeng Lyu at Rochester Institute of Technology.
