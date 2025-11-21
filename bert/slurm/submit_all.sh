#!/bin/bash
# Submit all Bert training tests to Slurm

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
mkdir -p logs

echo "Submitting Bert training tests to Slurm..."
echo ""

# Submit Bert training test
echo "Submitting Bert training test..."
JOB_BERT=$(sbatch --parsable slurm/bert.sbatch)
echo "  Job ID: $JOB_BERT"
echo ""

# Submit DistilBert training test
echo "Submitting DistilBert training test..."
JOB_DISTILBERT=$(sbatch --parsable slurm/distilbert.sbatch)
echo "  Job ID: $JOB_DISTILBERT"
echo ""