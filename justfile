# to install just: run this in powershell:
# winget install --id Casey.Just --exact

set shell:= ["pwsh", "-c"]

PROJECT_NAME:= "ibis_crew_ai"
REMOTE_REPO := "https://github.com/actuaristai/ibis-crew-ai.git"
DESCRIPTION := "Data manipulation using ibis with AI"



POWERSHELL_SHEBANG := if os() == 'windows' {
  'pwsh.exe'
} else {
  '/usr/bin/env pwsh'
}

###
### start with these commands
# just (will list out all the available just commands)
# just init-git (Only need to do once)
# just init-project (init-env, init-pre-commit, init-dvc)
# just lint (ruff)
# just test (pytest)
# just run (dvc repro)
# just docs (quarto)


# Initial help file
help:
	just --list --unsorted

# Regular commands to run
# ------------------------------------------

# build api documentation using quarto and render
_docs-build:
	uv run quartodoc build
	uv run quarto render

# build docs and preview using quarto
docs: _docs-build
	uv run quarto preview

# Lint using ruff
lint: 
	# uv run --extra lint codespell
	uv run --extra lint ruff check src/{{PROJECT_NAME}} --fix
	uv run --extra lint ruff check tests --fix
	uv run --extra lint ruff check src/frontend --fix

# test using pytest
test:
	#!/usr/bin/env bash
	uv run pytest tests/unit
	uv run pytest tests/integration
	# uv run --dev pytest --cov-report term-missing --cov={{PROJECT_NAME}} -v -p no:faulthandler -W ignore::DeprecationWarning --verbose --doctest-modules
	# uv run --dev pytest --cov-report term-missing --cov=tests -v -p no:faulthandler -W ignore::DeprecationWarning --verbose --doctest-modules

# set up playground environment
run:
	#!/usr/bin/env bash
	uv run uvicorn ibis_crew_ai.server:app --host localhost --port 8000 &
	uv run streamlit run src/frontend/streamlit_app.py --browser.serverAddress=localhost --server.enableCORS=false --server.enableXsrfProtection=false

# set up api 
backend: 
	uv run uvicorn ibis_crew_ai.server:app --host 0.0.0.0 --port 8000 --reload


# update template using copier. optional: use other copier options like vcs-ref=branch 
update-template *COPIER_OPTIONS:
	uvx copier update --trust --skip-tasks --skip-answered


# Initialisations - only need to be run once
# ------------------------------------------

# set up all to start up a project
_init-all: init-git init-project lint test _docs-build

# set up project (after cloning existing repository)
init-project: init-env init-pre-commit init-dvc

# initialise git. can alter REMOTE_REPO argument
init-git:
	git init --initial-branch=develop && git remote add origin {{REMOTE_REPO}}
	git add .
	git commit -m 'feat: Initial commit'

# create github repository and push initial git to remote
init-git-push:
	gh repo create {{PROJECT_NAME}} --public --homepage https://actuaristai.github.io/{{PROJECT_NAME}} --description "{{DESCRIPTION}}"
	git add .
	git commit -m 'feat: add dvc and qmd initialisations'
	git push -u origin develop
	git checkout -b main
	git push -u origin main

# set up environment using uv and set up ipykernel in project name
init-env:
	uv sync --dev --extra streamlit --extra jupyter --frozen
	uv run python -m ipykernel install --user --name {{PROJECT_NAME}}

# pre-commit
init-pre-commit:
	uvx pre-commit install --hook-type pre-commit --hook-type commit-msg
	uvx pre-commit autoupdate
	uvx pre-commit run --all-files

# set up dvc
init-dvc:
	uv run dvc init
	@echo "To setup dvc remote, enter DVC_SECRET in environment or .secrets.toml and run: just init-dvc-remote"


# set up dvc remote. ensure DVC_SECRET is in environment or in .secrets.toml file
init-dvc-remote DVC_REMOTE_NAME DVC_REMOTE DVC_SECRET:
	#!{{POWERSHELL_SHEBANG}}
	echo "initializing dvc into {{DVC_REMOTE}}"
	uv run dvc remote add -d {{DVC_REMOTE_NAME}} --local {{DVC_REMOTE}}
	uv run dvc remote modify {{DVC_REMOTE_NAME}} --local connection_string '{{DVC_SECRET}}'

# Initialise blank gh-pages branch for publishing
init-gh-pages:
	git checkout --orphan gh-pages
	echo n | git reset --hard # make sure all changes are committed before running this! # no as it asks do delete erroneous docs directory
	git commit --allow-empty -m "feat: Initialising gh-pages branch"  --no-verify
	git push origin gh-pages
	git checkout develop

# Continuous deployment functions
# ------------------------------------------

# publish to github pages
cd-publish:
	# bug workaround for lack of pre-commit in gh-pages branch
	$env:PRE_COMMIT_ALLOW_NO_CONFIG = "1"; uv run quarto publish gh-pages

# release version with tag (only for maintainers with merge permissions). Usage: just cd-release 'yyyy.mm.dd'
cd-release VERSION:
	git checkout -b release-{{VERSION}} develop
	uv run python bump_version.py {{VERSION}}
	uv sync
	uv run cz changelog --incremental
	git commit -a -m "chore: Bumped version number to {{VERSION}}"
	git checkout main
	git merge --no-ff release-{{VERSION}}
	git push
	git tag -a {{VERSION}} -m "add version tag"
	git push origin {{VERSION}}
	git checkout develop
	git merge --no-ff main
	git branch -d release-{{VERSION}}
	git push

# Other - adhoc useful commands
# ------------------------------------------

# clean. dash in front of command to ignore errors
clean:
	#!{{POWERSHELL_SHEBANG}}
	Remove-Item -Path "_freeze" -Recurse -Confirm -Erroraction 'silentlycontinue'
	Remove-Item -Path ".pytest_cache" -Recurse -Confirm -Erroraction 'silentlycontinue'
	Remove-Item -Path ".ruff_cache" -Recurse -Confirm -Erroraction 'silentlycontinue' 
	Remove-Item -Path "__pycache__" -Recurse -Confirm -Erroraction 'silentlycontinue'
	Remove-Item -Path ".quarto" -Recurse -Confirm -Erroraction 'silentlycontinue'
	Get-ChildItem -Path . -Filter "__pycache__" -Recurse -Directory | Remove-Item -Recurse -Force

# deploy api to google cloud run
cd-deploy:
	gcloud run deploy genai-app-sample --source . --project 'ace-world-453411-e9' --region 'us-central1' --memory "4Gi" --allow-unauthenticated

gcloud-auth:
	gcloud auth application-default login
	gcloud auth application-default set-quota-project ace-world-453411-e9
	gcloud config set project ace-world-453411-e9

# set up dev environment using terraform
setup-dev-env:
	#!/usr/bin/env bash
	export PROJECT_ID=$(gcloud config get-value project)
	(cd deployment/terraform/dev && terraform init && terraform apply --var-file vars/env.tfvars --var dev_project_id=$PROJECT_ID --auto-approve)
