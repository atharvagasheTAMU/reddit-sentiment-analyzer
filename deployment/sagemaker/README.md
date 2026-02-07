# SageMaker Real-Time Inference

This folder provides a minimal template to deploy the summarization + sentiment model to an AWS SageMaker real-time endpoint.

## 1) Package the model

Bundle your model artifacts in a `model.tar.gz` with the following structure:

```
model.tar.gz
  └── model/
      ├── config.json
      ├── pytorch_model.bin
      └── tokenizer.json (and other tokenizer files)
```

Upload the `model.tar.gz` to S3 and keep the URI.

## 2) Deploy

Set environment variables and run the deploy script:

```
cd deployment/sagemaker
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

$env:SAGEMAKER_ROLE_ARN="arn:aws:iam::123456789012:role/YourSageMakerRole"
$env:MODEL_ARTIFACT_S3_URI="s3://your-bucket/model.tar.gz"
$env:ENDPOINT_NAME="reddit-analyxer-realtime"
$env:INSTANCE_TYPE="ml.g4dn.xlarge"

python deploy.py
```

## 3) Invoke

Send JSON with a `text` field to `/invocations`:

```
{"text": "Your Reddit post content..."}
```

## Optional: Sarcasm Detection

If you have a sarcasm model, set `SARCASM_MODEL` as an environment variable at deploy time.

