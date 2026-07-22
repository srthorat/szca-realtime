#!/bin/bash
# SZCA Complete Test Runner
# Runs all test suites: Rust Gateway, C++ Engine, Integration, E2E, Performance, Security
#
# M17 FIXES:
#   * `set -o pipefail` so a failing command in a pipeline is not masked by a
#     trailing `tail`/`awk`. We deliberately do NOT `set -e` because we want to
#     run every suite and aggregate results; instead each suite's real exit
#     code is captured explicitly.
#   * No hard-coded pass counts. Pass/fail is decided purely by real exit codes.
#   * "ALL TESTS PASSED" is printed only when every suite exited 0, and the
#     script exits non-zero if anything failed.

set -uo pipefail

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  SZCA Test Suite v5.0.0                                     ║"
echo "║  Running all tests...                                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

FAILURES=0
PASSED_SUITES=0
FAILED_SUITES=""
SKIPPED_SUITES=""

# run_suite <label> <command...>
# Runs the command, streams output, captures the command's REAL exit code
# (pipefail ensures the tee pipeline reflects the command, not tee).
run_suite() {
  local label="$1"
  shift
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  ${label}"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  local rc
  "$@"
  rc=$?

  if [ "$rc" -eq 0 ]; then
    echo "  ✅ ${label} passed"
    PASSED_SUITES=$((PASSED_SUITES + 1))
  else
    echo "  ❌ ${label} FAILED (exit code ${rc})"
    FAILURES=$((FAILURES + 1))
    FAILED_SUITES="${FAILED_SUITES}\n    - ${label}"
  fi
  echo ""
  return 0
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# 1. Rust Gateway Unit Tests  (szca_media_gateway)
# ============================================================================
run_suite "[1/6] Rust Gateway Unit Tests" \
  bash -c "cd '${REPO_ROOT}/szca_media_gateway' && cargo test"

# ============================================================================
# 2. C++ Engine Unit Tests  (ctest in szca_onnx_engine/build)
# ============================================================================
CPP_BUILD_DIR="${REPO_ROOT}/szca_onnx_engine/build"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [2/6] C++ Engine Unit Tests (ctest)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -d "$CPP_BUILD_DIR" ]; then
  ( cd "$CPP_BUILD_DIR" && ctest --output-on-failure )
  rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "  ✅ C++ Engine Unit Tests passed"
    PASSED_SUITES=$((PASSED_SUITES + 1))
  else
    echo "  ❌ C++ Engine Unit Tests FAILED (exit code ${rc})"
    FAILURES=$((FAILURES + 1))
    FAILED_SUITES="${FAILED_SUITES}\n    - [2/6] C++ Engine Unit Tests"
  fi
else
  echo "  ⚠️  C++ build dir not found ($CPP_BUILD_DIR)."
  echo "     Build first: cd szca_onnx_engine && ./build.sh"
  SKIPPED_SUITES="${SKIPPED_SUITES}\n    - [2/6] C++ Engine Unit Tests (not built)"
fi
echo ""

# ============================================================================
# 3-6. Rust integration / e2e / performance / security  (szca_tests)
# ============================================================================
run_suite "[3/6] Integration Tests" \
  bash -c "cd '${REPO_ROOT}/szca_tests' && cargo test integration"

run_suite "[4/6] E2E Tests" \
  bash -c "cd '${REPO_ROOT}/szca_tests' && cargo test e2e"

run_suite "[5/6] Performance Tests" \
  bash -c "cd '${REPO_ROOT}/szca_tests' && cargo test benchmark"

run_suite "[6/6] Security Tests" \
  bash -c "cd '${REPO_ROOT}/szca_tests' && cargo test security"

# ============================================================================
# Summary
# ============================================================================
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  TEST SUMMARY                                               ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "  Suites passed: ${PASSED_SUITES}"
echo "  Suites failed: ${FAILURES}"
if [ -n "$FAILED_SUITES" ]; then
  echo -e "  Failed suites:${FAILED_SUITES}"
fi
if [ -n "$SKIPPED_SUITES" ]; then
  echo -e "  Skipped suites:${SKIPPED_SUITES}"
fi
echo ""

if [ "$FAILURES" -eq 0 ] && [ -z "$SKIPPED_SUITES" ]; then
  echo "  🎉 ALL TESTS PASSED!"
  exit 0
elif [ "$FAILURES" -eq 0 ]; then
  echo "  ⚠️  All executed suites passed, but some were SKIPPED (see above)."
  exit 1
else
  echo "  ⚠️  Some tests failed. Review output above."
  exit 1
fi
