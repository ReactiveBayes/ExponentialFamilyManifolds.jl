SHELL = /bin/bash
.DEFAULT_GOAL = help

.PHONY: lint format

scripts_init:
	julia --project=scripts/ -e 'using Pkg; Pkg.instantiate(); Pkg.update(); Pkg.precompile();'

lint: scripts_init ## Code formating check
	julia --project=scripts/ scripts/format.jl

format: scripts_init ## Code formating run
	julia --project=scripts/ scripts/format.jl --overwrite
	
help:  ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)