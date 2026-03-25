variable "DOCKERHUB_REPO" {
  default = "nortus"
}

variable "DOCKERHUB_IMG" {
  default = "comfyui-qwen-template"
}

variable "RELEASE_VERSION" {
  default = "latest"
}

target "default" {
  context = "."
  dockerfile = "Dockerfile"
  target = "final"
  platforms = ["linux/amd64"]
  tags = ["${DOCKERHUB_REPO}/${DOCKERHUB_IMG}:${RELEASE_VERSION}"]
}
