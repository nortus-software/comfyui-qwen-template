variable "DOCKERHUB_REPO" {
  default = "nortus"
}

variable "DOCKERHUB_IMG" {
  default = "comfyui-qwen-template"
}

variable "RELEASE_VERSION" {
  default = "latest"
}

group "default" {
  targets = ["cu128", "cu130"]
}

target "_common" {
  context    = "."
  dockerfile = "Dockerfile"
  target     = "final"
  platforms  = ["linux/amd64"]
}

# CUDA 12.8 — broad host compatibility (driver 570+). Tagged as :latest.
target "cu128" {
  inherits = ["_common"]
  args = {
    CUDA_VERSION   = "12.8.0"
    TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu128"
  }
  tags = [
    "${DOCKERHUB_REPO}/${DOCKERHUB_IMG}:${RELEASE_VERSION}",
    "${DOCKERHUB_REPO}/${DOCKERHUB_IMG}:${RELEASE_VERSION}-cu128",
  ]
}

# CUDA 13.0 — requires driver 580+. Nightly torch wheels (cu130 not yet stable). Opt-in tag.
target "cu130" {
  inherits = ["_common"]
  args = {
    CUDA_VERSION   = "13.0.0"
    TORCH_INDEX_URL = "https://download.pytorch.org/whl/nightly/cu130"
  }
  tags = [
    "${DOCKERHUB_REPO}/${DOCKERHUB_IMG}:${RELEASE_VERSION}-cu130",
  ]
}
