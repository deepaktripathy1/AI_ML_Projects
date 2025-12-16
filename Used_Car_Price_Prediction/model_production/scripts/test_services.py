"""Test script to generate metrics for monitoring."""

import requests
import time
import random
from datetime import datetime

# Service URLs
FASTAPI_URL = "http://localhost:8000"
BENTOML_URL = "http://localhost:3001"

# Sample car data
sample_cars = [
    {
        "make_year": 2020,
        "mileage_kmpl": 15.5,
        "engine_cc": 1500,
        "accidents_reported": 0,
        "fuel_type": "Petrol",
        "owner_count": 1,
        "brand": "Toyota",
        "transmission": "Automatic",
        "color": "White",
        "insurance_valid": "Yes",
        "service_history": "Full Service History"
    },
    {
        "make_year": 2018,
        "mileage_kmpl": 18.2,
        "engine_cc": 1200,
        "accidents_reported": 1,
        "fuel_type": "Diesel",
        "owner_count": 2,
        "brand": "Honda",
        "transmission": "Manual",
        "color": "Black",
        "insurance_valid": "Yes",
        "service_history": "Partial Service History"
    },
    {
        "make_year": 2022,
        "mileage_kmpl": 20.0,
        "engine_cc": 1800,
        "accidents_reported": 0,
        "fuel_type": "Electric",
        "owner_count": 1,
        "brand": "Tesla",
        "transmission": "Automatic",
        "color": "Silver",
        "insurance_valid": "Yes",
        "service_history": "Full Service History"
    }
]


def test_health_endpoints():
    """Test health endpoints."""
    print("\n" + "="*50)
    print("Testing Health Endpoints")
    print("="*50)

    # Test FastAPI
    try:
        response = requests.get(f"{FASTAPI_URL}/health", timeout=5)
        print(f"✓ FastAPI Health: {response.status_code}")
        print(f"  Response: {response.json()}")
    except Exception as e:
        print(f"✗ FastAPI Health Failed: {e}")

    # Test BentoML
    try:
        response = requests.get(f"{BENTOML_URL}/health", timeout=5)
        print(f"✓ BentoML Health: {response.status_code}")
        print(f"  Response: {response.json()}")
    except Exception as e:
        print(f"✗ BentoML Health Failed: {e}")


def test_metrics_endpoints():
    """Test metrics endpoints."""
    print("\n" + "="*50)
    print("Testing Metrics Endpoints")
    print("="*50)

    # Test FastAPI metrics
    try:
        response = requests.get(f"{FASTAPI_URL}/metrics", timeout=5)
        print(f"✓ FastAPI Metrics: {response.status_code}")
        lines = response.text.split('\n')
        print(
            f"  Total metrics: {len([l for l in lines if l and not l.startswith('#')])}")
    except Exception as e:
        print(f"✗ FastAPI Metrics Failed: {e}")

    # Test BentoML metrics
    try:
        response = requests.get(f"{BENTOML_URL}/metrics", timeout=5)
        print(f"✓ BentoML Metrics: {response.status_code}")
        lines = response.text.split('\n')
        print(
            f"  Total metrics: {len([l for l in lines if l and not l.startswith('#')])}")
    except Exception as e:
        print(f"✗ BentoML Metrics Failed: {e}")


def generate_predictions(service="fastapi", count=10):
    """Generate predictions to create metrics."""
    print(f"\n" + "="*50)
    print(f"Generating {count} predictions for {service.upper()}")
    print("="*50)

    url = f"{FASTAPI_URL}/predict" if service == "fastapi" else f"{BENTOML_URL}/predict_single"

    success_count = 0
    error_count = 0

    for i in range(count):
        car_data = random.choice(sample_cars)

        try:
            response = requests.post(url, json=car_data, timeout=10)

            if response.status_code == 200:
                result = response.json()
                price = result.get('predicted_price', 'N/A')
                confidence = result.get('confidence_score', 'N/A')
                print(
                    f"  [{i+1}/{count}] ✓ Price: ${price:.2f}, Confidence: {confidence}")
                success_count += 1
            else:
                print(f"  [{i+1}/{count}] ✗ Status: {response.status_code}")
                error_count += 1

        except Exception as e:
            print(f"  [{i+1}/{count}] ✗ Error: {str(e)[:50]}")
            error_count += 1

        # Small delay between requests
        time.sleep(0.5)

    print(f"\nSummary: {success_count} successful, {error_count} failed")


def continuous_load_test(duration_seconds=60):
    """Run continuous predictions for load testing."""
    print(f"\n" + "="*50)
    print(f"Running continuous load test for {duration_seconds} seconds")
    print("="*50)

    start_time = time.time()
    request_count = 0

    while (time.time() - start_time) < duration_seconds:
        car_data = random.choice(sample_cars)
        service = random.choice(["fastapi", "bentoml"])

        url = f"{FASTAPI_URL}/predict" if service == "fastapi" else f"{BENTOML_URL}/predict_single"

        try:
            response = requests.post(url, json=car_data, timeout=5)
            request_count += 1

            if request_count % 10 == 0:
                elapsed = time.time() - start_time
                rate = request_count / elapsed
                print(f"  Requests: {request_count}, Rate: {rate:.2f} req/s")

        except Exception as e:
            print(f"  Error: {str(e)[:50]}")

        time.sleep(random.uniform(0.1, 0.5))

    total_time = time.time() - start_time
    avg_rate = request_count / total_time
    print(f"\nCompleted: {request_count} requests in {total_time:.2f}s")
    print(f"Average rate: {avg_rate:.2f} requests/second")


def main():
    """Main test runner."""
    print("\n" + "="*70)
    print(" ML Model Monitoring - Service Test Suite")
    print("="*70)
    print(f" Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    # Test health
    test_health_endpoints()

    # Test metrics
    test_metrics_endpoints()

    # Generate some predictions
    print("\n--- FastAPI Predictions ---")
    generate_predictions(service="fastapi", count=5)

    print("\n--- BentoML Predictions ---")
    generate_predictions(service="bentoml", count=5)

    # Ask user about load test
    print("\n" + "="*50)
    response = input("Run continuous load test? (y/n): ")

    if response.lower() == 'y':
        duration = input("Duration in seconds (default 60): ")
        try:
            duration = int(duration) if duration else 60
        except:
            duration = 60
        continuous_load_test(duration)

    print("\n" + "="*70)
    print(" Test Suite Completed!")
    print("="*70)
    print("\nNext steps:")
    print("1. Check Prometheus targets: http://localhost:9090/targets")
    print("2. View Grafana dashboard: http://localhost:3000")
    print("3. Explore metrics in Prometheus: http://localhost:9090/graph")
    print("="*70)


if __name__ == "__main__":
    main()
