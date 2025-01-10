# Check that given variables are set and all have non-empty values,
# die with an error otherwise.
#
# Params:
#   1. Variable name(s) to test.
#   2. (optional) Error message to print.
check_defined = \
    $(strip $(foreach 1,$1, \
        $(call __check_defined,$1,$(strip $(value 2)))))
__check_defined = \
    $(if $(value $1),, \
      $(error Undefined variable: $1$(if $2, ($2))))

release:
	@:$(call check_defined, version, The release version)
	git checkout main
	git fetch
	git pull
	git tag -a $(version) -m "$(version) release"
	git push --tags origin $(version)

build-app:
	@:$(call check_defined, name, The app name)
	pmbuilder image --appdir ./atm/apps/$(name) --pipconf pip.conf --no-sandbox --local-git-build

test-app:
	@:$(call check_defined, name, The app name) 
	pmbuilder test image --appdir ./atm/apps/$(name)
