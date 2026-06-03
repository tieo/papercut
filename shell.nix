# Dev shell for NixOS: makes uv-installed binary wheels (numpy, scipy,
# scikit-learn, xgboost, torch, etc.) importable by surfacing the system
# libraries they expect via LD_LIBRARY_PATH.
#
# Usage:
#   nix-shell --run "uv sync --dev"
#   nix-shell --run "uv run pytest"
#
# Direnv users can drop `use nix` into .envrc and skip the explicit wrapping.
{ pkgs ? import <nixpkgs> { } }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    stdenv.cc.cc.lib
    zlib
  ];

  shellHook = ''
    export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [
      pkgs.stdenv.cc.cc.lib
      pkgs.zlib
    ]}:$LD_LIBRARY_PATH
  '';
}
