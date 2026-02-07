import os

import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel


def main() -> None:
    role = os.environ["SAGEMAKER_ROLE_ARN"]
    model_artifact = os.environ["MODEL_ARTIFACT_S3_URI"]
    region = os.environ.get("AWS_REGION", "us-east-1")

    session = sagemaker.Session(boto3_session=boto3.Session(region_name=region))

    model = PyTorchModel(
        model_data=model_artifact,
        role=role,
        entry_point="inference.py",
        source_dir="code",
        framework_version="2.1.0",
        py_version="py310",
        sagemaker_session=session,
        env={
            "SENTIMENT_MODEL": os.getenv(
                "SENTIMENT_MODEL", "distilbert-base-uncased-finetuned-sst-2-english"
            ),
            "SARCASM_MODEL": os.getenv("SARCASM_MODEL", ""),
        },
    )

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=os.getenv("INSTANCE_TYPE", "ml.g4dn.xlarge"),
        endpoint_name=os.getenv("ENDPOINT_NAME", "reddit-analyxer-realtime"),
    )
    print(predictor.endpoint_name)


if __name__ == "__main__":
    main()

