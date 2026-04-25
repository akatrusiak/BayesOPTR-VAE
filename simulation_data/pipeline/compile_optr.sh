#!/bin/bash
# compile_optr.sh — Build the TRANSOPTR executable only.
# Stripped from runoptr.sh: no running, no plotting, no gnuplot.
#
# Usage:
#   ./compile_optr.sh [OPTR_SRC_DIR] [WORK_DIR]
#
#   OPTR_SRC_DIR: directory containing the TRANSOPTR src/ folder (default: env var OPTRDIR)
#   WORK_DIR:     directory containing sy.f and data.dat (default: current directory)
#
# The script:
#   1. Cleans old .o files in src/
#   2. Compiles all src/*.f
#   3. Compiles sy.f in the working directory
#   4. Links everything into an 'optr' executable
#   5. Cleans up sy.o
#
# After running this once, the Python pipeline calls pyoptr.run() which
# invokes ./optr directly — no recompilation needed between runs.

set -e  # exit on error

# ── Check dependencies ────────────────────────────────────────────
hash gfortran 2>/dev/null || { echo >&2 "gfortran compiler not found"; exit 1; }

# ── Arguments ─────────────────────────────────────────────────────
OPTRDIR="${1:-$OPTRDIR}"
WORKDIR="${2:-$(pwd)}"

if [ -z "$OPTRDIR" ]; then
    echo "Usage: compile_optr.sh OPTR_SRC_DIR [WORK_DIR]"
    echo "  or set OPTRDIR environment variable"
    exit 1
fi

if [ ! -d "$OPTRDIR/src" ]; then
    echo "Error: $OPTRDIR/src not found"
    exit 1
fi

if [ ! -f "$WORKDIR/sy.f" ]; then
    echo "Error: $WORKDIR/sy.f not found"
    exit 1
fi

# ── Compiler flags ────────────────────────────────────────────────
FFLAGS="-std=legacy -finit-local-zero"

# Optional: double precision (uncomment if needed)
# FFLAGS="$FFLAGS -fdefault-real-8 -fdefault-integer-8"

# Optional: optimization (recommended for production runs)
FFLAGS="$FFLAGS -O3"

# ── Compile source library ────────────────────────────────────────
echo "Compiling TRANSOPTR source in $OPTRDIR/src/ ..."
cd "$OPTRDIR/src"
rm -f *.o 2>/dev/null
gfortran $FFLAGS -c *.f
echo "  Source compiled: $(ls *.o | wc -l) object files"

# ── Compile sy.f and link ─────────────────────────────────────────
cd "$WORKDIR"

# Clean old artifacts (but NOT data.dat)
rm -f fort.1 fort.8 fort.16 fort.14 fort.15 fort.envelope fort.xml \
      fort.label fort.8* fort.9* fort.12 fort.22 fort.3 \
      default.gnu envelope.gnu fort.1040 fort.13 2>/dev/null

echo "Compiling sy.f in $WORKDIR ..."
gfortran $FFLAGS -c sy.f

echo "Linking optr executable ..."
rm -f optr 2>/dev/null
gfortran $FFLAGS -o optr sy.o "$OPTRDIR"/src/*.o

# Clean up
rm -f sy.o 2>/dev/null

if [ ! -f optr ]; then
    echo "Error: optr executable was not created"
    exit 1
fi

echo "Done. Executable: $WORKDIR/optr"
