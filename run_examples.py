from pathlib import Path
from hyperon import MeTTa

# Run the MeTTa file
# metta_file = Path(__file__).parent / "examples" / "test_spectral.metta"
metta_file = Path(__file__).parent / "examples" / "test_spectral_adaptive_code_change.metta"

if metta_file.exists():
    with open(metta_file) as f:
        results = MeTTa().run(f.read())
    for result in results:
        print(result)
else:
    print(f"File not found: {metta_file}")