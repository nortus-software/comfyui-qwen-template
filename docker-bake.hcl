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

variable "WORKFLOW_VERSION" {
  default = "1"
}

target "serverless" {
  context = "."
  dockerfile = "Dockerfile.serverless"
  target = "final"
  platforms = ["linux/amd64"]
  tags = ["${DOCKERHUB_REPO}/${DOCKERHUB_IMG}:${RELEASE_VERSION}-serverless"]
  args = {
    WORKFLOW_VERSION = "${WORKFLOW_VERSION}"
  }
  secret = [
    "id=GITHUB_PAT,env=GITHUB_PAT",
    "id=HF_TOKEN,env=HF_TOKEN"
  ]
}
