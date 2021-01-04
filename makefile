.PHONY: all git_process

all: git_process

git_process:
	@git add --all
	@git commit -m "fast commit"
	@git push
