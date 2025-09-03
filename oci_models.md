# Inference serving language models in OCI-compliant model containers

You can inference serve OCI-compliant models in Red Hat AI Inference Server.
Storing models in OCI-compliant models containers (modelcars) is an alternative to S3 or URI-based storage for language models.

Using modelcar containers allows for faster startup times by avoiding repeated downloads, lower disk usage, and better performance with pre-fetched images.
Before you can deploy a language model in a modelcar in the cluster, you need to package the model in an OCI container image and then deploy the container image in the cluster.

* [Using OCI containers for model storage](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/latest/html-single/serving_models/index#using-oci-containers-for-model-storage_serving-large-models)

# Creating a modelcar image and pushing it to a container image registry

You can create a modelcar image that contains a language model that you can deploy with Red Hat AI Inference Server.

To create a modelcar image, download the model from [Hugging Face](https://huggingface.co/) and then package it into a container image and push the modelcar container to an image registry.

* You have installed Python 3.11 or later.
* You have installed Podman or Docker.
* You have access to the internet to download models from Hugging Face.
* You have configured a container image registry that you can push images to and have logged in.

1. Create a Python virtual environment and install the `huggingface_hub` Python library:

    ```bash
    python3 -m venv venv && \
    source venv/bin/activate && \
    pip install --upgrade pip && \
    pip install huggingface_hub
    ```
2. Create a model downloader Python script:

    ```bash
    vi download_model.py
    ```
3. Add the following content to the `download_model.py` file, adjusting the value for `model_repo` as required:

    ```python
    from huggingface_hub import snapshot_download

    # Specify the Hugging Face repository containing the model
    model_repo = "ibm-granite/granite-3.1-2b-instruct"
    snapshot_download(
        repo_id=model_repo,
        local_dir="/models",
        allow_patterns=["*.safetensors", "*.json", "*.txt"],
    )
    ```
4. Create a `Dockerfile` for the modelcar:

    ```dockerfile
    FROM registry.access.redhat.com/ubi9/python-311:latest as base

    USER root

    RUN pip install huggingface-hub

    # Download the model file from Hugging Face
    COPY download_model.py .

    RUN python download_model.py

    # Final image containing only the essential model files
    FROM registry.access.redhat.com/ubi9/ubi-micro:9.4

    # Copy the model files from the base container
    COPY --from=base /models /models

    USER 1001
    ```
5. Build the modelcar image:

    ```terminal
    podman build . -t modelcar-example:latest --platform linux/amd64
    ```

    **Example output**

    ```terminal
    Successfully tagged localhost/modelcar-example:latest
    ```
6. Push the modelcar image to the container registry. For example:

    ```terminal
    $ podman push modelcar-example:latest quay.io/<your_model_registry>/modelcar-example:latest
    ```

    **Example output**

    ```terminal
    Getting image source signatures
    Copying blob b2ed7134f853 done
    Copying config 4afd393610 done
    Writing manifest to image destination
    Storing signatures
    ```

# Inference serving modelcar images with AI Inference Server in OpenShift Container Platform

Deploy a language model in a modelcar container with OpenShift Container Platform by configuring secrets, persistent storage, and a deployment custom resource (CR) that uses Red Hat AI Inference Server to inference serve the modelcar container image.

* You have installed the OpenShift CLI (`oc`).
* You have logged in as a user with `cluster-admin` privileges.
* You have installed NFD and the required GPU Operator for your underlying AI accelerator hardware.
* You have created a modelcar container image for the language model and pushed it to a container image registry.

1. Set the cluster namespace to match where you deployed the Red Hat AI Inference Server image, for example:

    ```terminal
    $ NAMESPACE=rhaiis-namespace
    ```
    1. Create a `PersistentVolumeClaim` (`PVC`) custom resource (CR) and apply it in the cluster.
    The following example `PVC` CR uses a default IBM VPC Block persistence volume.

        ```yaml
        apiVersion: v1
        kind: PersistentVolumeClaim
        metadata:
          name: model-cache
          namespace: rhaiis-namespace
        spec:
          accessModes:
            - ReadWriteOnce
          resources:
            requests:
              storage: 20Gi
          storageClassName: ibmc-vpc-block-10iops-tier
        ```

        <dl><dt><strong>📌 NOTE</strong></dt><dd>

        Configuring cluster storage to meet your requirements is outside the scope of this procedure.
        For more detailed information, see [Configuring persistent storage](https://docs.redhat.com/en/documentation/openshift_container_platform/latest/html/storage/configuring-persistent-storage).
        </dd></dl>
    2. Create a `Deployment` custom resource (CR) that pulls the modelcar image and deploys the Red Hat AI Inference Server container.
    Reference the following example `Deployment` CR, which uses AI Inference Server to serve a modelcar image.

        ```yaml
        apiVersion: apps/v1
        kind: Deployment
        metadata:
          name: rhaiis-oci-deploy
          namespace: rhaiis-namespace
          labels:
            app: granite
        spec:
          replicas: 0
          selector:
            matchLabels:
              app: rhaiis-oci-deploy
          template:
            metadata:
              labels:
                app: rhaiis-oci-deploy
            spec:
              imagePullSecrets:
                - name: docker-secret
              volumes:
                - name: model-volume
                  persistentVolumeClaim:
                    claimName: model-cache ①
                - name: shm
                  emptyDir:
                    medium: Memory
                    sizeLimit: "2Gi"
                - name: oci-auth
                  secret:
                    secretName: docker-secret
                    items:
                      - key: .dockercfg
                        path: config.json
              initContainers: ②
                - name: fetch-model
                  image: ghcr.io/oras-project/oras:v1.2.0
                  command: ["/bin/sh","-c"]
                  args:
                    - |
                      set -e
                      # Only pull if /model is empty
                      if [ -z "$(ls -A /model)" ]; then
                        echo "Pulling model…"
                        oras pull <your_modelcar_registry_url> \ ③
                          --output /model \
                      else
                        echo "Model already present, skipping pull"
                      fi
                  volumeMounts:
                    - name: model-volume
                      mountPath: /model
                    - name: oci-auth
                      mountPath: /auth
                      readOnly: true
              containers:
                - name: granite
                  image: 'registry.redhat.io/rhaiis/vllm-cuda-rhel9@sha256:a6645a8e8d7928dce59542c362caf11eca94bb1b427390e78f0f8a87912041cd'
                  imagePullPolicy: IfNotPresent
                  env:
                    - name: VLLM_SERVER_DEV_MODE
                      value: '1'
                  command:
                    - python
                    - '-m'
                    - vllm.entrypoints.openai.api_server
                  args:
                    - '--port=8000'
                    - '--model=/model'
                    - '--served-model-name=ibm-granite/granite-3.1-2b-instruct' ④
                    - '--tensor-parallel-size=1'
                  resources:
                    limits:
                      cpu: '10'
                      nvidia.com/gpu: '1'
                      memory: 16Gi
                    requests:
                      cpu: '2'
                      memory: 6Gi
                      nvidia.com/gpu: '1'
                  volumeMounts:
                    - name: model-volume
                      mountPath: /model
                    - name: shm
                      mountPath: /dev/shm ⑤
              restartPolicy: Always
        ```
        1. `spec.template.spec.volumes.persistentVolumeClaim.claimName` must match the name of the `PVC` that you created.
        2. This example deployment uses a simple `initContainers` configuration that runs before the main app container to download the required modelcar image.
        The model pull step is skipped if the model directory has already been populated, for example, from a previous deployment.
        3. The image registry URL for the modelcar image that you want to inference.
        4. Update the value for `--served-model-name` to match the model that you are deploying.
        5. The `/dev/shm` volume mount is required by the NVIDIA Collective Communications Library (NCCL).
        Tensor parallel vLLM deployments fail when the `/dev/shm` volume mount is not set.
            1. Increase the deployment replica count to the required number. For example, run the following command:

                ```terminal
                oc scale deployment granite -n rhaiis-namespace --replicas=1
                ```
            2. Optional: Watch the deployment and ensure that it succeeds:

                ```terminal
                $ oc get deployment -n rhaiis-namespace --watch
                ```

                **Example output**

                ```terminal
                NAME                READY   UP-TO-DATE   AVAILABLE   AGE
                rhaiis-oci-deploy   0/1     1            0           2s
                rhaiis-oci-deploy   1/1     1            1           14s
                ```
    3. Create a `Service` CR for the model inference. For example:

        ```yaml
        apiVersion: v1
        kind: Service
        metadata:
          name: rhaiis-oci-deploy
          namespace: rhaiis-namespace
        spec:
          selector:
            app: rhaiis-oci-deploy
          ports:
            - name: http
              port: 80
              targetPort: 8000

        ```
    4. Optional: Create a `Route` CR to enable public access to the model. For example:

        ```yaml
        apiVersion: route.openshift.io/v1
        kind: Route
        metadata:
          name: rhaiis-oci-deploy
          namespace: rhaiis-namespace
        spec:
          to:
            kind: Service
            name: rhaiis-oci-deploy
          port:
            targetPort: http
        ```
    5. Get the URL for the exposed route. Run the following command:

        ```terminal
        $ oc get route granite -n rhaiis-namespace -o jsonpath='{.spec.host}'
        ```

        **Example output**

        ```terminal
        rhaiis-oci-deploy-rhaiis-namespace.apps.example.com
        ```

Ensure that the deployment is successful by querying the model.
Run the following command:

```terminal
curl -v -k   http://rhaiis-oci-deploy-aireilly-rhaiis.apps.modelsibm.ibmmodel.rh-ods.com/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model":"ibm-granite/granite-3.1-2b-instruct",
    "messages":[{"role":"user","content":"Hello?"}],
    "temperature":0.1
  }'| jq
```

**Example output**

```terminal
{
  "id": "chatcmpl-07b177360eaa40a3b311c24a8e3c7f43",
  "object": "chat.completion",
  "created": 1755189746,
  "model": "ibm-granite/granite-3.1-2b-instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "reasoning_content": null,
        "content": "Hello! How can I assist you today?",
        "tool_calls": []
      },
      "logprobs": null,
      "finish_reason": "stop",
      "stop_reason": null
    }
  ],
  "usage": {
    "prompt_tokens": 61,
    "total_tokens": 71,
    "completion_tokens": 10,
    "prompt_tokens_details": null
  },
  "prompt_logprobs": null,
  "kv_transfer_params": null
}
```
