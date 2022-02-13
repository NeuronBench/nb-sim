{ pkgs ? import <nixpkgs> {} }:

let
  mozilla-overlay = import (builtins.fetchTarball https://github.com/
pkgs.mkShell {
  packages = [ pkgs.rustc pkgs.cargo pkgs.cargo-watch pkgs.rust-analyzer ];
}
