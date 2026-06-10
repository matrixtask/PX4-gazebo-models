#!/usr/bin/env bash
set -euo pipefail

OPENVSP_HOME="${OPENVSP_HOME:-$HOME/tools/openvsp-3.50.5/opt/OpenVSP}"
OPENVSP_LIB_DIR="${OPENVSP_LIB_DIR:-$HOME/tools/openvsp-libs/root/usr/lib/x86_64-linux-gnu}"
MODEL_DIR="models/teTra_mk-7_EM2"
MODEL_BASENAME="teTra_mk-7_EM2"

export LD_LIBRARY_PATH="$OPENVSP_LIB_DIR:${LD_LIBRARY_PATH:-}"

usage() {
    cat <<USAGE
Usage: $0 [model|geom|sweep|stab|all|clean]

  model  Generate ${MODEL_DIR}/${MODEL_BASENAME}.vsp3
  geom   Generate VSPAERO geometry files from the .vsp3 model
  sweep  Run a minimal VSPAERO sweep to create solver setup/results
  stab   Run VSPAERO stability derivatives from the generated setup
  all    Run model, geom, sweep, and stab
  clean  Remove generated VSPAERO solver output files, keeping .vsp3/.stab
USAGE
}

run_model() {
    run_vsp_script "$MODEL_DIR/$MODEL_BASENAME.vsp3" \
        tools/tetra_mk7_openvsp_model.vspscript
}

run_vsp_script() {
    local expected_output="$1"
    shift

    if "$OPENVSP_HOME/vsp" -script "$@"; then
        return 0
    fi

    if [[ -f "$expected_output" ]]; then
        echo "OpenVSP returned nonzero after generating $expected_output; continuing."
        return 0
    fi

    return 1
}

run_geom() {
    run_vsp_script "$MODEL_DIR/$MODEL_BASENAME.vspgeom" \
        tools/tetra_mk7_vspaero_compute_geometry.vspscript
}

run_sweep() {
    run_vsp_script "$MODEL_DIR/$MODEL_BASENAME.vspaero" \
        tools/tetra_mk7_vspaero_sweep.vspscript
}

run_stab() {
    ( cd "$MODEL_DIR" && "$OPENVSP_HOME/vspaero" -stab "$MODEL_BASENAME" )
}

clean_outputs() {
    rm -f "$MODEL_DIR"/"$MODEL_BASENAME".adb \
          "$MODEL_DIR"/"$MODEL_BASENAME".adb.cases \
          "$MODEL_DIR"/"$MODEL_BASENAME".case.*.quad.1.dat \
          "$MODEL_DIR"/"$MODEL_BASENAME".csf \
          "$MODEL_DIR"/"$MODEL_BASENAME".cuts \
          "$MODEL_DIR"/"$MODEL_BASENAME".flt \
          "$MODEL_DIR"/"$MODEL_BASENAME".group.* \
          "$MODEL_DIR"/"$MODEL_BASENAME".history \
          "$MODEL_DIR"/"$MODEL_BASENAME".lod \
          "$MODEL_DIR"/"$MODEL_BASENAME".polar \
          "$MODEL_DIR"/"$MODEL_BASENAME".quad.cases \
          "$MODEL_DIR"/"$MODEL_BASENAME".slc \
          "$MODEL_DIR"/"$MODEL_BASENAME".vkey \
          "$MODEL_DIR"/"$MODEL_BASENAME".vspaero \
          "$MODEL_DIR"/"$MODEL_BASENAME".vspgeom
}

cmd="${1:-all}"
case "$cmd" in
    model) run_model ;;
    geom) run_geom ;;
    sweep) run_sweep ;;
    stab) run_stab ;;
    all)
        run_model
        run_geom
        run_sweep
        run_stab
        ;;
    clean) clean_outputs ;;
    -h|--help|help) usage ;;
    *)
        usage
        exit 2
        ;;
esac
