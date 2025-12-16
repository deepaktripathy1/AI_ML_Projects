import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.feature_store.hopsworks_client import HopsworksClient
from src.feature_store.feast_client import FeastClient
from src.utils.path_utils import from_serializable_path
from config.config import get_settings


def create_reference_data():
    """Create a sample reference dataset for drift detection."""
    print("Creating reference data for drift detection...")

    settings = get_settings()

    # Create a data directory if it doesn"t exist
    data_dir = settings.data_dir
    data_dir.mkdir(exist_ok=True)
    monitoring_dir = data_dir / "monitoring"
    monitoring_dir.mkdir(exist_ok=True)

    # Check if processed data path exists
    processed_data_path = monitoring_dir / "processed_data.csv"
    if processed_data_path.exists():
        print(f"Processed data already exists at {processed_data_path}")
        return

    # Check if reference data already exists
    reference_data_path = data_dir / "monitoring" / "reference_data.csv"
    if reference_data_path.exists():
        print(f"Reference data already exists at {reference_data_path}")
        return

    # Check for Hopswork availability and get training data
    try:
        response = requests.get(
            f"https://{settings.HOPSWORKS_HOST}",
            timeout=10,
            verify=True
        )

        if response.status_code == 200:
            print("Hopsworks up and running")
            client = HopsworksClient()
            # Get feature view
            feature_view = client.get_feature_view(
                name="used_car_price_features",
                version=1
            )
            # Get training data
            if feature_view is not None:
                version, _ = feature_view.create_train_test_split(
                    test_size=settings.TEST_SIZE,
                    description="Create metadata for training dataset",
                    data_format="csv"
                )
                X_train, X_test, y_train, y_test = feature_view.get_train_test_split(
                    training_dataset_version=version,
                    dataframe_type="pandas"
                )
                # Create the final dataframe
                if isinstance(y_train, pd.Series):
                    y_train = y_train.to_frame(settings.TARGET_COLUMN)
                if isinstance(y_test, pd.Series):
                    y_test = y_test.to_frame(settings.TARGET_COLUMN)

                train_df = pd.concat([X_train, y_train], axis=1)
                test_df = pd.concat([X_test, y_test], axis=1)

                df = pd.concat([train_df, test_df], axis=0).reset_index(
                    drop=True
                )
                # Remove metadata columns
                df.drop(columns=["car_id"], inplace=True)
                df.to_csv(
                    from_serializable_path(str(processed_data_path)), index=False
                )
                print("Processed dataset loaded from Hopsworks")

        else:
            print("Hopsworks unavailable")

    except Exception as e:
        print(f"Failed to connect to Hopsworks: {e}")

        # Connect to Feast store
        client = FeastClient()
        feature_view = client.get_feature_view(
            name="used_car_price_features",
            version=1
        )

        # Load data
        feature_view_name = "used_car_price_features"
        version = 1
        full_name = f"{feature_view_name}_v{version}"
        data_path = client.data_path / f"{full_name}.parquet"
        if not data_path.exists():
            raise FileNotFoundError(
                f"Feature data not found at {data_path}"
            )
        df = pd.read_parquet(data_path)
        df.drop(
            columns=["car_id", "event_timestamp"],
            inplace=True
        )
        # Convert to csv
        df.to_csv(from_serializable_path(
            str(processed_data_path)), index=False)

    np.random.seed(42)
    n_samples = 1000

    # Load processed data
    df = pd.read_csv(from_serializable_path(str(processed_data_path)))

    # Create a sample of processed data as reference data
    reference_data = df.sample(n=n_samples)

    # bool columns
    bool_cols = [
        "has_accident",
        "fuel_type_electric",
        "fuel_type_petrol",
        "brand_chevrolet",
        "brand_ford",
        "brand_honda",
        "brand_hyundai",
        "brand_kia",
        "brand_nissan",
        "brand_tesla",
        "brand_toyota",
        "brand_volkswagen",
        "transmission_manual",
        "color_blue",
        "color_gray",
        "color_red",
        "color_silver",
        "color_white",
        "insurance_valid_yes"
    ]

    for col in bool_cols:
        if col in reference_data.columns:
            reference_data[col] = reference_data[col].astype(int)

    # Save reference data
    reference_data.to_csv(
        from_serializable_path(str(reference_data_path)),
        index=False
    )

    print(f"Created reference data: {reference_data_path}")
    print(
        f"  Dataset contains {n_samples} samples with {
            len(reference_data.columns)
        } features"
    )


def validate_monitoring_setup():
    """Validate that the monitoring setup is correct."""
    print("\nValidating monitoring setup...")

    checks = [
        ("Monitoring directory", lambda: Path("monitoring").exists()),
        ("Prometheus config", lambda: Path("monitoring/prometheus").exists()),
        ("Grafana config", lambda: Path("monitoring/grafana").exists()),
        ("AlertManager config", lambda: Path("monitoring/alertmanager").exists()),
        (
            "Environment file",
            lambda: Path(".env.monitoring").exists() or Path(".env").exists(),
        ),
        ("Source monitoring modules", lambda: Path("src/monitoring").exists()),
        ("Data directory", lambda: Path("data").exists()),
        ("Logs directory", lambda: Path("logs").exists()),
    ]

    all_passed = True
    for check_name, check_func in checks:
        try:
            if check_func():
                print(f"✅ {check_name}")
            else:
                print(f"❌ {check_name}")
                all_passed = False
        except Exception as e:
            print(f"❌ {check_name}: {e}")
            all_passed = False

    return all_passed


def start_monitoring_services():
    """Start monitoring services using Docker Compose."""
    print("\nStarting monitoring services...")

    try:
        # Check if Docker is available
        subprocess.check_call(
            ["docker", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        subprocess.check_call(
            ["docker-compose", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Start monitoring stack
        result = subprocess.run(
            ["docker-compose", "-f", "docker-compose.yml", "up", "-d"],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print("Monitoring services started successfully")
            print("\nServices available at:")
            print("  • Prometheus: http://localhost:9090")
            print("  • Grafana: http://localhost:3000 (admin/admin123)")
            print("  • AlertManager: http://localhost:9093")
            return True
        else:
            print(f"Failed to start monitoring services: {result.stderr}")
            return False

    except subprocess.CalledProcessError:
        print("Docker or Docker Compose not found. Please install Docker first.")
        return False


def print_integration_instructions():
    """Print instructions for integrating monitoring with existing FastAPI app."""
    print("\n" + "=" * 60)
    print("MONITORING INTEGRATION INSTRUCTIONS")
    print("=" * 60)

    print("\n1. Install monitoring dependencies (done above)")

    print("\n2. Update your FastAPI app to include monitoring:")
    print("   • Import monitoring middleware in your main.py:")
    print("     from src.monitoring.monitoring_middleware import MonitoringMiddleware")
    print("     app.add_middleware(MonitoringMiddleware)")

    print("\n3. Update your prediction endpoints:")
    print("   • Import metrics tracker:")
    print("     from src.monitoring.metrics_tracker import track_prediction")
    print("   • Add tracking to your prediction function")

    print("\n4. Set up drift detection:")
    print("   • The reference data has been created in data/reference_data.csv")
    print("   • Update MONITORING_REFERENCE_DATA_PATH in your .env file if needed")

    print("\n5. Configure alerts:")
    print("   • Update MONITORING_ALERT_EMAIL_RECIPIENTS in your .env file")
    print("   • Add MONITORING_SLACK_WEBHOOK_URL if using Slack notifications")

    print("\n6. Start monitoring services:")
    print("   docker-compose -f docker-compose.monitoring.yml up -d")

    print("\n7. Test the setup:")
    print("   • Make predictions via your API")
    print("   • Check metrics at http://localhost:3001 (Grafana)")
    print("   • Trigger drift check: POST /monitoring/drift/check")


def main():
    """Main setup function."""
    print("Setting up ML Model Monitoring for Existing Project...")
    print("=" * 55)

    try:
        # Create reference data for drift detection
        create_reference_data()

        # Validate setup
        if validate_monitoring_setup():
            print("\nMonitoring setup validation passed")
        else:
            print("\nSome validation checks failed")

        # Print integration instructions
        print_integration_instructions()

        # Ask if user wants to start services
        print("\n" + "=" * 60)
        response = input(
            "\nWould you like to start the monitoring services now? (y/n): "
        )
        if response.lower() in ["y", "yes"]:
            start_monitoring_services()
        else:
            print("\nYou can start the monitoring services later with:")
            print("docker-compose -f docker-compose.monitoring.yml up -d")

    except Exception as e:
        print(f"\nSetup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
