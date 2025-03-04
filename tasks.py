from invoke import task
import os
import sys
import subprocess
from pathlib import Path

@task
def setup_venv(ctx, env_path=".venv"):
    """
    Set up the Biotuner environment automatically.

    :param env_path: Path to create the virtual environment.
                     Default is '.venv' inside the Biotuner repo.
    """

    # Force the script to always operate from biotuner/
    repo_root = Path(__file__).resolve().parent
    os.chdir(repo_root)  # Hard enforce working directory

    print(f"📁 Enforcing repo root: {repo_root}")
    env_full_path = repo_root / env_path
    print(f"📌 Expected venv path: {env_full_path}")

    # DELETE any incorrectly created venv before continuing
    wrong_venv_path = repo_root.parent / ".venv"
    if wrong_venv_path.exists():
        print(f"⚠️ Removing incorrectly created venv in {wrong_venv_path}...")
        subprocess.run(["rm", "-rf", str(wrong_venv_path)], check=True, shell=True)

    # Create virtual environment **inside biotuner/**
    if not env_full_path.exists():
        print(f"🛠️ Creating virtual environment at {env_full_path} with Python {sys.version.split()[0]}...")
        subprocess.run([sys.executable, "-m", "venv", str(env_full_path)], check=True)
    else:
        print(f"✅ Virtual environment '{env_full_path}' already exists.")

    # Ensure `requirements.txt` exists
    requirements_file = repo_root / "requirements.txt"
    if not requirements_file.exists():
        print(f"❌ Error: {requirements_file} not found in {repo_root}.")
        sys.exit(1)

    # Install dependencies **inside the virtual environment**
    print(f"📦 Installing dependencies from {requirements_file}...")

    python_exec = str(env_full_path / "Scripts" / "python") if os.name == "nt" else str(env_full_path / "bin" / "python")

    subprocess.run([python_exec, "-m", "pip", "install", "-r", str(requirements_file)], check=True)

    # Install Biotuner in editable mode **inside the correct directory**
    print("⚙️ Installing Biotuner in editable mode...")
    subprocess.run([python_exec, "-m", "pip", "install", "-e", str(repo_root)], check=True)

    print("🎉 Setup complete! To activate your environment, run:")
    if os.name == "nt":
        print(f"🔹 {env_full_path}\\Scripts\\activate")
    else:
        print(f"🔹 source {env_full_path}/bin/activate")

@task
def setup(ctx, env_name="biotuner_env"):
    """
    Set up the Biotuner environment automatically using Conda.
    
    :param env_name: Name of the Conda environment to create.
                     Default is 'biotuner_env'.
    """
    print(f"📁 Setting up Conda environment: {env_name}")

    # Check if Conda is installed
    try:
        subprocess.run(["conda", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        print("❌ Conda is not installed. Please install Miniconda or Anaconda.")
        return

    # Check if the Conda environment already exists
    existing_envs = subprocess.run(["conda", "env", "list"], check=True, capture_output=True, text=True)
    if env_name in existing_envs.stdout:
        print(f"✅ Conda environment `{env_name}` already exists.")
    else:
        # Create Conda environment
        print(f"🛠️ Creating Conda environment `{env_name}`...")
        subprocess.run(["conda", "create", "-n", env_name, "python=3.11", "-y"], check=True)
        print(f"✅ Created Conda environment `{env_name}`.")

    # Activate the Conda environment and install dependencies
    print(f"📦 Installing dependencies in `{env_name}`...")
    subprocess.run(["conda", "run", "-n", env_name, "pip", "install", "-r", "requirements.txt"], check=True)

    # Install Biotuner in editable mode
    print("⚙️ Installing Biotuner in editable mode...")
    subprocess.run(["conda", "run", "-n", env_name, "pip", "install", "-e", "."], check=True)

    print("🎉 Setup complete! To activate your environment, run:")
    print(f"🔹 conda activate {env_name}")

@task
def test(ctx):
    """
    Run all tests inside the `tests` directory.
    """
    repo_root = Path(__file__).resolve().parent
    tests_path = repo_root / "tests"

    if not tests_path.exists():
        print(f"❌ Error: Tests directory not found at {tests_path}")
        return

    print(f"🚀 Running tests in {tests_path}...")

    # Run pytest
    result = subprocess.run(["pytest", str(tests_path)], check=False)

    if result.returncode == 0:
        print("✅ All tests passed successfully!")
    else:
        print("❌ Some tests failed. Check the output above.")

