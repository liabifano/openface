THIS_DIR := $(dir $(abspath $(firstword $(MAKEFILE_LIST))))
IMAGES_PATH="$(THIS_DIR)/resources/photos"
VECTORS_PATH="$(THIS_DIR)/resources"
VOLUME_IMAGES_PATH="/photos"
VOLUME_VECTORS_PATH="/vectors"

help:
	@echo "clone-openface    - clone openface repository"
	@echo "build-openface    - build openface docker"
	@echo "bash-openface     - adhoc openface docker run"
	@echo "get-photos        - query and get photos from slack and save in /resources/photos"
	@echo "get-vectors       - get face vectors for collected photos and save in /resources/vectors.json"
	@echo "bootstrap         - clone-openface && build-openface && get-photos && get-vectors"
	@echo "run               - run application"


clone-openface:
	@bash -c "git clone https://github.com/liabifano/openface"

build-openface:
	@echo "...... Go get a coffee, it will take some time ;)"
	@bash -c "cd openface/ && docker build -t openface ."

bash-openface:
	@docker run --entrypoint "/bin/bash" -v $(IMAGES_PATH):$(VOLUME_IMAGES_PATH) -v $(VECTORS_PATH):$(VOLUME_VECTORS_PATH) -p 5000:5000 -it openface

get-vectors:
	@docker run -v $(IMAGES_PATH):$(VOLUME_IMAGES_PATH) -v $(VECTORS_PATH):$(VOLUME_VECTORS_PATH) -it openface python /root/openface/app/app.py get-vectors

get-photos:
	@bash -c "echo WIP"

bootstrap:
	@bash -c "make clone-openface && make build-openface && make get-photos && make get-vectors"

run:
	@docker run -v $(IMAGES_PATH):$(VOLUME_IMAGES_PATH) -v $(VECTORS_PATH):$(VOLUME_VECTORS_PATH) -p 5000:5000 -it openface python /root/openface/app/app.py run
