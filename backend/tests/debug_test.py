"""
Manual Supabase connectivity check.

Run directly when you want to verify real environment variables and network
access:

    python backend/tests/debug_test.py

This file is intentionally not a pytest test because it requires real
Supabase credentials and a live project.
"""

from pathlib import Path
import sys

from supabase import Client, create_client


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))


def main() -> None:
    from config import get_settings

    print("--- 1. TERMINAL CHECK ---")
    print(f"Current working directory: {Path.cwd()}")
    print(f"Backend directory: {BACKEND_DIR}")
    print()

    print("--- 2. PYDANTIC SETTINGS PARSING ---")

    try:
        settings = get_settings()

        print("Pydantic settings parsed successfully.")
        print(f"APP_NAME: {settings.app_name}")
        print(f"SUPABASE_URL: {settings.supabase_url}")
        print(f"DATABASE_API_KEY found: {bool(settings.database_api_key)}")
        print(f"AUTH_API_KEY found: {bool(settings.auth_api_key)}")
        print(f"GEMINI_API_KEY found: {bool(settings.gemini_api_key)}")
    except Exception as exc:
        print(f"Pydantic validation failed: {exc}")
        raise SystemExit(1)

    print()
    print("--- 3. SUPABASE CLIENT CONNECTION ---")

    try:
        supabase: Client = create_client(
            settings.supabase_url,
            settings.database_api_key,
        )

        print("Supabase client initialized successfully.")
    except Exception as exc:
        print(f"Supabase client initialization failed: {exc}")
        raise SystemExit(1)

    print()
    print("--- 4. DOCUMENTS QUERY TEST ---")

    try:
        response = (
            supabase
            .table("documents")
            .select("id")
            .limit(1)
            .execute()
        )

        print("Supabase API query succeeded.")
        print(f"Rows returned: {response.data}")

    except Exception as exc:
        print(f"Network/API error: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
