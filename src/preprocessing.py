import pandas as pd
import holidays
from pathlib import Path
import logging
import time

# -------------------------------------------------------
# Configuração de paths
# -------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_PATH = BASE_DIR / "data/raw"
PROCESSED_PATH = BASE_DIR / "data/processed"
LOG_PATH = BASE_DIR / "logs"

YEAR = 2015

LOG_PATH.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------
# Configuração do logger (CSV)
# -------------------------------------------------------

logger = logging.getLogger("pipeline")
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(LOG_PATH / "pipeline_logs.csv")

formatter = logging.Formatter(
    "%(asctime)s,%(levelname)s,%(funcName)s,%(message)s"
)

file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

# -------------------------------------------------------
# Decorator para medir tempo das etapas
# -------------------------------------------------------

def log_step(func):
    def wrapper(*args, **kwargs):

        start = time.time()

        logger.info(f"START {func.__name__}")

        result = func(*args, **kwargs)

        elapsed = time.time() - start

        if isinstance(result, pd.DataFrame):
            logger.info(
                f"END {func.__name__},rows={len(result):,},time_sec={elapsed:.2f}"
            )
        else:
            logger.info(
                f"END {func.__name__},time_sec={elapsed:.2f}"
            )

        return result

    return wrapper


# -------------------------------------------------------
# Load data
# -------------------------------------------------------

@log_step
def load_data():

    flight_columns = [
        "YEAR","MONTH","DAY","DAY_OF_WEEK",
        "AIRLINE","FLIGHT_NUMBER","TAIL_NUMBER",
        "ORIGIN_AIRPORT","DESTINATION_AIRPORT",
        "SCHEDULED_DEPARTURE","DEPARTURE_DELAY",
        "SCHEDULED_TIME","DISTANCE","SCHEDULED_ARRIVAL",
        "CANCELLED","DIVERTED"
    ]

    flight_dtypes = {
        "YEAR": "int16",
        "MONTH": "int8",
        "DAY": "int8",
        "DAY_OF_WEEK": "int8",
        "AIRLINE": "category",
        "FLIGHT_NUMBER": "int32",
        "TAIL_NUMBER": "category",
        "ORIGIN_AIRPORT": "category",
        "DESTINATION_AIRPORT": "category",
        "SCHEDULED_DEPARTURE": "int16",
        "DEPARTURE_DELAY": "float32",
        "SCHEDULED_TIME": "float32",
        "DISTANCE": "int16",
        "SCHEDULED_ARRIVAL": "int16",
        "CANCELLED": "int8",
        "DIVERTED": "int8"
    }

    flights = pd.read_csv(
        RAW_PATH / "flights.csv",
        usecols=flight_columns,
        dtype=flight_dtypes
    )

    airlines = pd.read_csv(RAW_PATH / "airlines.csv")

    airports = pd.read_csv(
        RAW_PATH / "airports.csv",
        usecols=["IATA_CODE","AIRPORT","CITY","STATE","LATITUDE","LONGITUDE"]
    )

    logger.info(f"Flights loaded {len(flights):,}")

    return flights, airlines, airports


# -------------------------------------------------------
# Merge datasets
# -------------------------------------------------------

@log_step
def merge_data(flights, airlines, airports):

    # origem
    df = flights.merge(
        airports.rename(columns={
            "LATITUDE": "ORIGIN_LAT",
            "LONGITUDE": "ORIGIN_LON",
            "AIRPORT": "ORIGIN_AIRPORT_NAME",
            "CITY": "ORIGIN_CITY",
            "STATE": "ORIGIN_STATE"
        }).set_index("IATA_CODE"),
        left_on="ORIGIN_AIRPORT",
        right_index=True,
        how="left"
    )

    # destino
    df = df.merge(
        airports.rename(columns={
            "LATITUDE": "DEST_LAT",
            "LONGITUDE": "DEST_LON",
            "AIRPORT": "DEST_AIRPORT_NAME",
            "CITY": "DEST_CITY",
            "STATE": "DEST_STATE"
        }).set_index("IATA_CODE"),
        left_on="DESTINATION_AIRPORT",
        right_index=True,
        how="left"
    )

    # airline
    df = df.merge(
        airlines.set_index("IATA_CODE"),
        left_on="AIRLINE",
        right_index=True,
        how="left"
    )

    df = df.rename(columns={
        "AIRLINE_x": "AIRLINE_CODE",
        "AIRLINE_y": "AIRLINE_NAME"
    })

    return df


# -------------------------------------------------------
# Filtering
# -------------------------------------------------------

@log_step
def filter_cancelled_diverted(df):

    df = df[
        (df["CANCELLED"] == 0) &
        (df["DIVERTED"] == 0)
    ]

    df = df.drop(columns=["CANCELLED","DIVERTED"], errors="ignore")

    return df


# -------------------------------------------------------
# Missing values
# -------------------------------------------------------

@log_step
def remove_missing_location(df):

    df = df.dropna(
        subset=["ORIGIN_LAT","ORIGIN_LON","DEST_LAT","DEST_LON"]
    )

    return df


# -------------------------------------------------------
# HHMM → minutos
# -------------------------------------------------------

@log_step
def hhmm_to_minutes(df):

    df["SCHEDULED_DEPARTURE_MIN"] = df["SCHEDULED_DEPARTURE"].apply(
        lambda hhmm: (int(hhmm) // 100) * 60 + (int(hhmm) % 100)
    )

    return df


# -------------------------------------------------------
# Feature engineering
# -------------------------------------------------------

@log_step
def create_features(df):

    df["DATE"] = pd.to_datetime(df[["YEAR","MONTH","DAY"]])
    df["HOUR"] = df["SCHEDULED_DEPARTURE"] // 100

    df["IS_DELAYED"] = (df["DEPARTURE_DELAY"] >= 15).astype("int8")

    df["IS_SHORT_DISTANCE"] = (df["DISTANCE"] < 500).astype("int8")
    df["IS_MEDIUM_DISTANCE"] = (
        (df["DISTANCE"] >= 500) &
        (df["DISTANCE"] < 1500)
    ).astype("int8")

    df["IS_LONG_DISTANCE"] = (df["DISTANCE"] >= 1500).astype("int8")

    df["IS_MORNING"] = df["HOUR"].between(0,11).astype("int8")
    df["IS_AFTERNOON"] = df["HOUR"].between(12,17).astype("int8")
    df["IS_NIGHT"] = df["HOUR"].between(18,23).astype("int8")

    us_holidays = holidays.US(years=YEAR)
    holiday_dates = pd.to_datetime(list(us_holidays.keys()))

    df["IS_HOLIDAY"] = df["DATE"].isin(holiday_dates).astype("int8")

    return df


# -------------------------------------------------------
# Top N categorias
# -------------------------------------------------------

@log_step
def create_top_n_feature(
    df,
    column,
    target="IS_DELAYED",
    top_n=10,
    new_column_name=None,
    drop_original=False
):

    if new_column_name is None:
        new_column_name = f"{column}_TOP{top_n}"

    filtered = df[df[target] == 1]

    counts = filtered[column].value_counts()

    top_categories = counts.head(top_n).index

    if isinstance(df[column].dtype, pd.CategoricalDtype):
        df[column] = df[column].cat.add_categories("Other")

    df[new_column_name] = df[column].where(
        df[column].isin(top_categories),
        "Other"
    ).astype("string")

    if drop_original:
        df = df.drop(columns=[column])

    return df



# -------------------------------------------------------
# Save data
# -------------------------------------------------------

@log_step
def save_data(df):

    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

    output_path = PROCESSED_PATH / "flights_processed.parquet"

    df.to_parquet(output_path, index=False)

    logger.info(f"Saved to {output_path}")


# -------------------------------------------------------
# Pipeline
# -------------------------------------------------------

def main():

    flights, airlines, airports = load_data()

    df = merge_data(flights, airlines, airports)

    df = filter_cancelled_diverted(df)

    df = hhmm_to_minutes(df)

    df = remove_missing_location(df)

    df = create_features(df)

    df = create_top_n_feature(df, "ORIGIN_AIRPORT")

    df = create_top_n_feature(df, "DESTINATION_AIRPORT")

    df = create_top_n_feature(df, "ORIGIN_CITY")

    df = create_top_n_feature(df, "DEST_CITY")

    df = create_top_n_feature(df, "ORIGIN_STATE")

    df = create_top_n_feature(df, "DEST_STATE")

    df = create_top_n_feature(df, "TAIL_NUMBER")

    df = create_top_n_feature(df, "AIRLINE")

    df = create_top_n_feature(df, "FLIGHT_NUMBER")


    df["ROUTE"] = df["ORIGIN_AIRPORT"].astype(str) + "-" + df["DESTINATION_AIRPORT"].astype(str)

    df = create_top_n_feature(df, "ROUTE")

    save_data(df)

    logger.info("Pipeline finished successfully")


if __name__ == "__main__":
    main()
