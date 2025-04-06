# Variables for AWS Microservices Infrastructure

variable "aws_region" {
  description = "The AWS region to deploy resources in"
  type        = string
  default     = "us-west-2"
}

variable "project_name" {
  description = "The name of the project"
  type        = string
  default     = "microservices-demo"
}

variable "availability_zones" {
  description = "The availability zones to deploy resources in"
  type        = list(string)
  default     = ["us-west-2a", "us-west-2b", "us-west-2c"]
}

variable "ecr_repository_url" {
  description = "The URL of the ECR repository where container images are stored"
  type        = string
  default     = "123456789012.dkr.ecr.us-west-2.amazonaws.com"
}