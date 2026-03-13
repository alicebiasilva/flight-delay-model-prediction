import pandas as pd
import holidays
from pathlib import Path
import logging

# -------------------------------------------------------
# Configuração
# -------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_PATH = BASE_DIR / "data/raw"
PROCESSED_PATH = BASE_DIR / "data/processed"

YEAR = 2015

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -------------------------------------------------------
# Load data
# -------------------------------------------------------

def load_data():

    logging.info("Loading raw datasets...")

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

    logging.info(f"Flights loaded: {len(flights):,}")

    return flights, airlines, airports


# -------------------------------------------------------
# Merge datasets
# -------------------------------------------------------

def merge_data(flights, airlines, airports):

    logging.info("Merging datasets...")

    df = flights.merge(
        airports.set_index("IATA_CODE"),
        left_on="ORIGIN_AIRPORT",
        right_index=True,
        how="left"
    )

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

    del flights

    return df


# -------------------------------------------------------
# Filtering
# -------------------------------------------------------

def filter_cancelled_diverted(df):

    logging.info("Filtering cancelled/diverted flights...")

    df = df[
        (df["CANCELLED"] == 0) &
        (df["DIVERTED"] == 0)
    ]

    return df


# -------------------------------------------------------
# Remove leakage
# -------------------------------------------------------

LEAKAGE_COLUMNS = [
    "CANCELLED",
    "DIVERTED",
    "AIRLINE_NAME"
]


def remove_data_leakage(df):

    logging.info("Removing leakage columns...")

    df = df.drop(columns=LEAKAGE_COLUMNS, errors="ignore")

    return df


# -------------------------------------------------------
# Missing values
# -------------------------------------------------------

def remove_missing_location(df):

    logging.info("Removing rows with missing location data...")

    df = df.dropna(
        subset=["AIRPORT","CITY","STATE","LATITUDE","LONGITUDE"]
    )

    return df


# -------------------------------------------------------
# Feature engineering
# -------------------------------------------------------

def create_features(df):

    logging.info("Creating features...")

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

    winter_dates = set(pd.date_range("2015-01-01","2015-03-19")) | \
                   set(pd.date_range("2015-12-21","2015-12-31"))

    spring_dates = set(pd.date_range("2015-03-20","2015-06-20"))
    summer_dates = set(pd.date_range("2015-06-21","2015-09-22"))
    fall_dates = set(pd.date_range("2015-09-23","2015-12-20"))

    df["IS_WINTER"] = df["DATE"].isin(winter_dates).astype("int8")
    df["IS_SPRING"] = df["DATE"].isin(spring_dates).astype("int8")
    df["IS_SUMMER"] = df["DATE"].isin(summer_dates).astype("int8")
    df["IS_FALL"] = df["DATE"].isin(fall_dates).astype("int8")

    return df


# -------------------------------------------------------
# Save data
# -------------------------------------------------------

def save_data(df):

    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

    output_path = PROCESSED_PATH / "flights_processed.parquet"

    logging.info(f"Saving dataset to {output_path}")

    df.to_parquet(output_path, index=False)

    del df


# -------------------------------------------------------
# Pipeline
# -------------------------------------------------------

def main():

    flights, airlines, airports = load_data()

    df = merge_data(flights, airlines, airports)

    df = filter_cancelled_diverted(df)

    df = remove_data_leakage(df)

    df = remove_missing_location(df)

    df = create_features(df)

    save_data(df)

    logging.info("Preprocessing finished successfully.")


if __name__ == "__main__":
    main()