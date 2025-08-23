{
  description = "Sakura - A high-performance minimal terminal-based multimedia library";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    systems.url = "github:nix-systems/default";
  };

  outputs = inputs @{ self, nixpkgs, systems, ... }:
    let
      inherit (nixpkgs) lib;

      eachSystem = lib.genAttrs (import systems);
      pkgsFor = eachSystem
        (system:
          import nixpkgs {
            localSystem.system = system;
          }
        );
    in
    {
      packages = lib.mapAttrs
        (system: pkgs:
          let
            fs = lib.fileset;
            src = fs.difference (fs.gitTracked ./.) (fs.unions [
              (fs.fileFilter (file: lib.strings.hasInfix ".git" file.name) ./.)
              (fs.fileFilter (file: file.hasExt "md") ./.)
              (fs.fileFilter (file: file.hasExt "nix") ./.)
            ]);
          in
          {
            default = self.packages.${system}.sakura;
            sakura = pkgs.gcc15Stdenv.mkDerivation {
              name = "sakura";
              src = fs.toSource {
                root = ./.;
                fileset = src;
              };

              nativeBuildInputs = with pkgs; [
                cmake
                pkg-config
              ];
              buildInputs = with pkgs; [
                opencv
                libsixel
                ffmpeg
                libcpr
                openssl
              ];

              cmakeFlags = [
                "--no-warn-unused-cli"
                "-O3"
                "-DNDEBUG"
                "-march=native"
                "-flto"
              ];
            };
          }
        )
        pkgsFor;
      nixosModules.default = { config, lib, pkgs, ... }:
        with lib; {
          options.programs.sakura = {
            enable = mkEnableOption "sakura";
          };

          config = mkIf config.programs.sakura.enable {
            environment.systemPackages = [
              self.packages.${pkgs.system}.sakura
            ];
          };
        };

      darwinModules.default = { config, lib, pkgs, ... }:
        with lib; {
          options.programs.sakura = {
            enable = mkEnableOption "sakura";
          };

          config = mkIf config.programs.sakura.enable {
            environment.systemPackages = [
              self.packages.${pkgs.system}.sakura
            ];
          };
        };

      homeModules.default = { config, lib, pkgs, ... }:
        with lib; {
          options.programs.sakura = {
            enable = mkEnableOption "sakura";

            package = mkOption {
              type = types.package;
              default = self.packages.${pkgs.system}.sakura;
              description = "The sakura package to use";
            };
          };

          config = mkIf config.programs.sakura.enable {
            home.packages = [ config.programs.sakura.package ];
          };
        };

      nixosModules.sakura = self.nixosModules.default;
      darwinModules.sakura = self.darwinModules.default;
      homeModules.sakura = self.homeModules.default;
    };
}
