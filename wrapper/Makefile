THIS_DIR := $(dir $(abspath $(firstword $(MAKEFILE_LIST))))
VOLUME_LOCAL_PATH="$(THIS_DIR)/app/static"
VOLUME_REMOTE_PATH="/root/openface/wrapper/app/static"

help:
	@echo "build-openface    - build openface docker"
	@echo "bash-openface     - adhoc openface docker run"
	@echo "get-vectors       - get face vectors for collected photos and save in /resources/vectors.json"
	@echo "bootstrap         - build-openface && get-vectors"
	@echo "run               - run application"


build-openface:
	@echo "...... Go grab a coffee, it will take some time ;)"
	@echo ""
	@bash -c "cd .. && docker build -t openface ."

bash-openface:
	@docker run --entrypoint "/bin/bash" -v $(VOLUME_LOCAL_PATH):$(VOLUME_REMOTE_PATH) -p 5000:5000 -it openface

get-vectors:
	@docker run -v $(VOLUME_LOCAL_PATH):$(VOLUME_REMOTE_PATH) -it openface python /root/openface/wrapper/app/app.py get-vectors

bootstrap:
	@bash -c "make clone-openface && make build-openface && make get-photos && make get-vectors"

run:
	@docker run -v $(VOLUME_LOCAL_PATH):$(VOLUME_REMOTE_PATH) -p 5000:5000 -it openface python /root/openface/wrapper/app/app.py run
